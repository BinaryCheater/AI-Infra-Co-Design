//! Current-edge compiled LSU/L1 controller and pinned Mem_SimWrapper boundary.
//! L2 remains a bounded directory/transaction abstraction, not inclusive-cache RTL.
use super::*;

#[derive(Clone, Copy)]
struct Request {
    token: Token,
    entry: usize,
    line: u64,
    write: bool,
}
struct Instruction {
    token: Token,
    remaining: usize,
    complete: bool,
    emitted: bool,
}
struct Address {
    token: Token,
    ready: Time,
    groups: VecDeque<u64>,
    entry: Option<usize>,
    shared: bool,
}
struct Primary {
    line: u64,
    targets: Vec<Request>,
    releasing: bool,
}
#[derive(Default)]
struct L1 {
    core: Option<Request>,
    primary: Vec<Option<Primary>>,
    lower: VecDeque<usize>,
    target: Option<Request>,
    stage: Option<Request>,
    responses: VecDeque<Request>,
    writes: HashSet<u64>,
}
#[derive(Clone)]
struct Lower {
    direct: bool,
    id: u64,
    line: u64,
    write: bool,
    owner: Option<(usize, usize)>,
    available: Time,
}
struct Fill {
    line: u64,
    way: usize,
    targets: Vec<Lower>,
}
#[derive(Clone, Copy)]
struct Response {
    id: u64,
    due: Time,
}
pub(super) struct External {
    slots: Vec<Option<Response>>,
    channels: usize,
    busy: Vec<Time>,
}
impl External {
    fn new(channels: usize) -> Self {
        Self {
            slots: vec![None; channels * 5],
            channels,
            busy: vec![0; channels],
        }
    }
    fn ready_slot(&self, line: u64, bytes: usize, now: Time) -> Option<usize> {
        let ch = line as usize / bytes % self.channels;
        if self.busy[ch] > now {
            return None;
        }
        (ch * 5..ch * 5 + 5).find(|i| self.slots[*i].is_none())
    }
    fn response(&self, now: Time) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.is_some_and(|s| s.due <= now))
    }
    fn insert(&mut self, slot: usize, id: u64, now: Time, delay: Time, interval: Time) {
        assert!(self.slots[slot].is_none());
        self.slots[slot] = Some(Response {
            id,
            due: now + delay + 1,
        });
        self.busy[slot / 5] = now + interval;
    }
}
pub(super) struct MemorySystem {
    address: Vec<Option<Address>>,
    used: Vec<Vec<Option<Instruction>>>,
    state: Vec<u8>,
    next: Vec<u8>,
    pre_complete: Vec<Vec<usize>>,
    pre_free: Vec<usize>,
    returned: Vec<bool>,
    l1: Vec<L1>,
    lds: Vec<VecDeque<(Time, Request)>>,
    lds_busy: Vec<Time>,
    lower: VecDeque<Lower>,
    fills: HashMap<u64, Fill>,
    transactions: HashMap<u64, Lower>,
    lower_done: VecDeque<(Time, Lower)>,
    external: External,
    outer: VecDeque<Lower>,
    source_d: Option<(Time, Lower)>,
    source_d_reuse: Time,
    serial: u64,
    pub audit: [u64; 12],
    clock: Time,
    flushed: bool,
}
impl MemorySystem {
    pub fn new(h: &[usize; 32]) -> Self {
        Self {
            address: (0..h[SMS]).map(|_| None).collect(),
            used: (0..h[SMS])
                .map(|_| (0..h[LSU]).map(|_| None).collect())
                .collect(),
            state: vec![0; h[SMS]],
            next: vec![0; h[SMS]],
            pre_complete: vec![Vec::new(); h[SMS]],
            pre_free: vec![0; h[SMS]],
            returned: vec![false; h[SMS]],
            l1: (0..h[SMS])
                .map(|_| L1 {
                    primary: (0..h[L1M]).map(|_| None).collect(),
                    ..L1::default()
                })
                .collect(),
            lds: vec![VecDeque::new(); h[SMS]],
            lds_busy: vec![0; h[SMS]],
            lower: VecDeque::new(),
            fills: HashMap::new(),
            transactions: HashMap::new(),
            lower_done: VecDeque::new(),
            external: External::new(h[CHANNELS]),
            outer: VecDeque::new(),
            source_d: None,
            source_d_reuse: 0,
            serial: 0,
            audit: [0; 12],
            clock: 0,
            flushed: false,
        }
    }
    fn enqueue_lower(
        &mut self,
        line: u64,
        write: bool,
        owner: Option<(usize, usize)>,
        now: Time,
    ) -> u64 {
        let id = self.serial;
        self.serial += 1;
        self.lower.push_back(Lower {
            direct: false,
            id,
            line,
            write,
            owner,
            available: now + 2,
        });
        id
    }
    pub fn drained(&self) -> bool {
        self.flushed
            && self.address.iter().all(Option::is_none)
            && self.used.iter().flatten().all(Option::is_none)
            && self.l1.iter().all(|c| {
                c.core.is_none()
                    && c.primary.iter().all(Option::is_none)
                    && c.lower.is_empty()
                    && c.target.is_none()
                    && c.stage.is_none()
                    && c.responses.is_empty()
                    && c.writes.is_empty()
            })
            && self.lds.iter().all(VecDeque::is_empty)
            && self.lower.is_empty()
            && self.fills.is_empty()
            && self.transactions.is_empty()
            && self.lower_done.is_empty()
            && self.external.slots.iter().all(Option::is_none)
            && self.outer.is_empty()
            && self.source_d.is_none()
    }
    fn flush(&mut self, now: Time, b: &mut Builder<'_>) {
        if self.lower.len() >= b.hw[L2M] {
            return;
        }
        // Fixed SM/set/line arbitration must not depend on host HashMap seeds.
        let victim = b
            .cache
            .l1
            .iter()
            .flat_map(|(key, q)| q.iter().filter(|t| t.dirty).map(|t| (*key, t.line)))
            .min();
        if let Some((key, line)) = victim {
            b.cache
                .l1
                .get_mut(&key)
                .unwrap()
                .iter_mut()
                .find(|t| t.line == line)
                .unwrap()
                .dirty = false;
            self.enqueue_lower(line, true, None, now);
        } else if let Some((key, line)) = b
            .cache
            .l2
            .iter()
            .flat_map(|(key, q)| q.iter().filter(|t| t.dirty).map(|t| (*key, t.line)))
            .min()
        {
            b.cache
                .l2
                .get_mut(&key)
                .unwrap()
                .iter_mut()
                .find(|t| t.line == line)
                .unwrap()
                .dirty = false;
            let id = self.enqueue_lower(line, true, None, now);
            self.lower.back_mut().unwrap().direct = true;
            assert_eq!(self.lower.back().unwrap().id, id);
        } else {
            self.flushed = true;
        }
    }
    pub fn retire(&mut self, t: Token) {
        if let Some(i) = self.used[t.sm]
            .iter()
            .position(|i| i.as_ref().is_some_and(|i| i.token.op == t.op))
        {
            let old = self.used[t.sm][i].take().unwrap();
            assert!(old.complete && old.emitted);
            if self.state[t.sm] == 2 {
                self.next[t.sm] = if self.pre_complete[t.sm]
                    .iter()
                    .any(|e| *e != self.pre_free[t.sm])
                {
                    2
                } else {
                    0
                };
            }
            self.audit[3] += 1;
        }
    }
    fn group_return(&mut self, r: Request) {
        let i = self.used[r.token.sm][r.entry]
            .as_mut()
            .expect("response without instruction MSHR");
        assert!(i.remaining > 0);
        i.remaining -= 1;
        i.complete = i.remaining == 0;
        self.audit[1] += 1;
    }
    pub fn step(&mut self, now: Time, b: &mut Builder<'_>, output_free: &[bool]) -> Vec<Token> {
        self.clock = now;
        let h = b.hw;
        let mut outputs = Vec::new();
        self.state.clone_from(&self.next);
        self.returned.fill(false);
        for sm in 0..h[SMS] {
            self.pre_complete[sm] = self.used[sm]
                .iter()
                .enumerate()
                .filter_map(|(e, i)| i.as_ref().filter(|i| i.complete).map(|_| e))
                .collect();
            self.pre_free[sm] = self.used[sm].iter().position(Option::is_none).unwrap_or(0);
            for i in self.used[sm].iter_mut().flatten() {
                i.emitted = false;
            }
            let old_complete = !self.pre_complete[sm].is_empty();
            // MSHRv2 output cannot consume a partial return or allocate on the
            // same edge. Requests/targets keep the instruction entry until WB.
            if self.state[sm] == 2 {
                if output_free[sm] {
                    if let Some(i) = self.used[sm]
                        .iter_mut()
                        .flatten()
                        .find(|i| i.complete && !i.emitted)
                    {
                        i.emitted = true;
                        outputs.push(Token {
                            time: now,
                            ..i.token
                        });
                    }
                }
                self.next[sm] = 0;
            } else if self.state[sm] == 1 {
                self.next[sm] = if old_complete && output_free[sm] {
                    2
                } else {
                    0
                };
            } else {
                let returned = self.l1[sm].responses.pop_front().or_else(|| {
                    if self.lds[sm].front().is_some_and(|(t, _)| *t <= now) {
                        self.lds[sm].pop_front().map(|(_, r)| r)
                    } else {
                        None
                    }
                });
                if let Some(r) = returned {
                    self.group_return(r);
                    self.returned[sm] = true;
                }
                self.next[sm] = if output_free[sm]
                    && self.used[sm]
                        .iter()
                        .any(|i| i.as_ref().is_some_and(|i| i.complete))
                {
                    2
                } else {
                    0
                };
            }
            let c = &mut self.l1[sm];
            let core_space = c.responses.len() < h[3];
            let stage_move = c.stage.is_some() && core_space;
            let target_move = c.target.is_some() && (c.stage.is_none() || stage_move);
            if stage_move {
                c.responses.push_back(c.stage.take().unwrap());
            }
            if target_move {
                c.stage = c.target.take();
            }
            let mut blocked =
                c.primary.iter().flatten().any(|p| p.releasing) || !c.lower.is_empty();
            if let Some(&pid) = c.lower.front() {
                blocked = true;
                if core_space {
                    let p = c.primary[pid].as_mut().unwrap();
                    if !p.releasing {
                        let set = (p.line / h[LINE] as u64) as usize % h[L1S];
                        let tags = b.cache.l1.entry((sm, set)).or_default();
                        if tags.len() >= h[L1W] && !tags.iter().any(|t| t.line == p.line) {
                            let victim = tags.pop_front().unwrap();
                            if victim.dirty {
                                self.enqueue_lower(victim.line, true, None, now);
                            }
                        }
                        let p = self.l1[sm].primary[pid].as_mut().unwrap();
                        let dirty = p.targets.iter().any(|r| r.write);
                        b.cache.l1.entry((sm, set)).or_default().push_back(Tag {
                            line: p.line,
                            fill: 0,
                            dirty,
                            way: 0,
                        });
                        p.releasing = true;
                    }
                    let c = &mut self.l1[sm];
                    let p = c.primary[pid].as_mut().unwrap();
                    if c.target.is_none() && !p.targets.is_empty() {
                        c.target = Some(p.targets.remove(0));
                    } else if p.targets.is_empty() && c.target.is_none() {
                        c.primary[pid] = None;
                        c.lower.pop_front();
                    }
                }
            }
            if blocked {
                self.audit[8] += 1;
                continue;
            }
            let Some(r) = self.l1[sm].core else {
                continue;
            };
            if self.l1[sm].target.is_some() {
                continue;
            }
            let set = (r.line / h[LINE] as u64) as usize % h[L1S];
            let tags = b.cache.l1.entry((sm, set)).or_default();
            if let Some(pos) = tags.iter().position(|t| t.line == r.line) {
                let mut tag = tags.remove(pos).unwrap();
                tag.dirty |= r.write;
                tags.push_back(tag);
                b.stats[L1_HIT] += 1;
                self.l1[sm].target = Some(r);
                self.l1[sm].core = None;
            } else if let Some(pid) = self.l1[sm]
                .primary
                .iter()
                .position(|p| p.as_ref().is_some_and(|p| p.line == r.line))
            {
                let p = self.l1[sm].primary[pid].as_mut().unwrap();
                if p.targets.len() < h[L1SUB] {
                    p.targets.push(r);
                    self.l1[sm].core = None;
                    b.stats[L1_MERGE] += 1;
                }
            } else if self.lower.len() < h[L2M] {
                if r.write && self.l1[sm].writes.len() < h[L1WRITE] {
                    let id = self.enqueue_lower(r.line, true, Some((sm, usize::MAX)), now);
                    self.l1[sm].writes.insert(id);
                    self.l1[sm].target = Some(r);
                    self.l1[sm].core = None;
                    b.stats[L1_WRITE_MISS] += 1;
                } else if !r.write {
                    if let Some(pid) = self.l1[sm].primary.iter().position(Option::is_none) {
                        self.l1[sm].primary[pid] = Some(Primary {
                            line: r.line,
                            targets: vec![r],
                            releasing: false,
                        });
                        self.enqueue_lower(r.line, false, Some((sm, pid)), now);
                        self.l1[sm].core = None;
                        b.stats[L1_READ_MISS] += 1;
                    }
                }
            }
        }
        self.step_lower(now, b);
        self.audit[10] = self.audit[10].max(self.external.slots.iter().flatten().count() as u64);
        outputs
    }
    fn lower_ready(&self, r: &Lower) -> bool {
        r.write || r.owner.is_none_or(|(sm, _)| self.l1[sm].lower.len() < 2)
    }
    fn deliver_lower(&mut self, r: Lower) {
        if let Some((sm, pid)) = r.owner {
            if r.write {
                assert!(self.l1[sm].writes.remove(&r.id));
            } else {
                self.l1[sm].lower.push_back(pid);
            }
        }
    }
    fn victim(b: &mut Builder<'_>) -> usize {
        let victim = (b.cache.lfsr as usize) & (b.hw[L2W] - 1);
        let x = b.cache.lfsr;
        let feedback = (x ^ (x >> 1) ^ (x >> 3) ^ (x >> 4)) & 1;
        b.cache.lfsr = if x == 0 { 1 } else { (x << 1) | feedback };
        victim
    }
    fn step_lower(&mut self, now: Time, b: &mut Builder<'_>) {
        let h = b.hw;
        // Queue boundaries: DCache memReq_Q + inner request stage; Directory
        // result registration + scheduler; SinkD response register + MSHR
        // scheduling; SourceD stage_1/stage_4, no same-edge busy reuse.
        let outer = self.outer.front().filter(|r| r.available <= now).cloned();
        let free = outer
            .as_ref()
            .and_then(|r| self.external.ready_slot(r.line, h[LINE], now));
        let source_free = self.source_d.is_none() && self.source_d_reuse <= now;
        if self
            .source_d
            .as_ref()
            .is_some_and(|(t, r)| *t <= now && self.lower_ready(r))
        {
            let (_, r) = self.source_d.take().unwrap();
            self.deliver_lower(r);
            self.source_d_reuse = now + 1;
        }
        if source_free && self.lower_done.front().is_some_and(|(t, _)| *t <= now) {
            let (_, r) = self.lower_done.pop_front().unwrap();
            self.source_d = Some((now + 1, r));
        }
        if let Some(index) = self.external.response(now) {
            let id = self.external.slots[index].unwrap().id;
            let r = self.transactions.get(&id).unwrap().clone();
            let targets = self.fills.get(&id).map(|f| f.targets.len()).unwrap_or(1);
            if self.lower_done.len() + targets <= h[L2M] && self.outer.len() < h[L2M] {
                self.external.slots[index] = None;
                self.transactions.remove(&id);
                self.audit[5] += 1;
                if r.write {
                    b.writes = b.writes.max(now);
                    self.lower_done.push_back((now + 3, r));
                } else {
                    let f = self.fills.remove(&id).unwrap();
                    let set = (f.line / h[LINE] as u64) as usize % h[L2S];
                    let tags = b.cache.l2.entry(set).or_default();
                    if let Some(pos) = tags.iter().position(|t| t.way == f.way) {
                        let old = tags.remove(pos).unwrap();
                        if old.dirty {
                            let id = self.serial;
                            self.serial += 1;
                            self.outer.push_back(Lower {
                                direct: true,
                                id,
                                line: old.line,
                                write: true,
                                owner: None,
                                available: now + 1,
                            });
                        }
                    }
                    tags.retain(|t| t.line != f.line);
                    tags.push_back(Tag {
                        line: f.line,
                        fill: 0,
                        dirty: false,
                        way: f.way,
                    });
                    for r in f.targets {
                        self.lower_done.push_back((now + 3, r));
                    }
                }
            } else {
                self.audit[9] += 1;
            }
        }
        if let (Some(r), Some(slot)) = (outer, free) {
            self.outer.pop_front();
            let delay = if (0x70000000..0x80000000).contains(&r.line) {
                0
            } else {
                h[MLAT] as u64
            };
            self.external
                .insert(slot, r.id, now, delay, h[LINE].div_ceil(h[BW]) as u64);
            b.stats[if r.write { MEMORY_WRITE } else { MEMORY_READ }] += h[LINE] as u64;
            self.audit[4] += 1;
            self.transactions.insert(r.id, r);
        } else if self.outer.front().is_some_and(|r| r.available <= now) {
            self.audit[11] += 1;
        }
        let Some(r) = self.lower.front().filter(|r| r.available <= now).cloned() else {
            return;
        };
        let set = (r.line / h[LINE] as u64) as usize % h[L2S];
        if !r.direct {
            if !r.write {
                if let Some(f) = self.fills.values_mut().find(|f| f.line == r.line) {
                    if f.targets.len() < h[L2M] {
                        f.targets.push(r);
                        self.lower.pop_front();
                        Self::victim(b);
                        b.stats[L2_HIT] += 1;
                    }
                    return;
                }
            }
            if b.cache
                .l2
                .get(&set)
                .is_some_and(|q| q.iter().any(|t| t.line == r.line))
            {
                if self.lower_done.len() < h[L2M] {
                    if r.write {
                        self.flushed = false;
                        b.cache
                            .l2
                            .get_mut(&set)
                            .unwrap()
                            .iter_mut()
                            .find(|t| t.line == r.line)
                            .unwrap()
                            .dirty = true;
                    }
                    self.lower.pop_front();
                    self.lower_done.push_back((now + 2, r));
                    Self::victim(b);
                    b.stats[L2_HIT] += 1;
                }
                return;
            }
        }
        let active: HashSet<_> = self
            .fills
            .keys()
            .chain(self.transactions.keys())
            .copied()
            .chain(self.outer.iter().map(|r| r.id))
            .collect();
        if active.len() >= h[L2M] || self.outer.len() >= h[L2M] {
            return;
        }
        self.lower.pop_front();
        let victim = Self::victim(b);
        if !r.write {
            self.fills.insert(
                r.id,
                Fill {
                    line: r.line,
                    way: victim,
                    targets: vec![r.clone()],
                },
            );
        }
        b.stats[L2_MISS] += 1;
        self.outer.push_back(Lower {
            available: now + 2,
            ..r
        });
    }
    fn admit(&mut self, sm: usize, now: Time, b: &mut Builder<'_>, ops: &[Op]) {
        let h = b.hw;
        let Some(mut a) = self.address[sm].take() else {
            return;
        };
        if now < a.ready {
            self.address[sm] = Some(a);
            return;
        }
        if a.entry.is_none() {
            if self.state[sm] != 0 {
                self.address[sm] = Some(a);
                return;
            }
            let Some(entry) = self.used[sm].iter().position(Option::is_none) else {
                self.address[sm] = Some(a);
                return;
            };
            self.used[sm][entry] = Some(Instruction {
                token: a.token,
                remaining: a.groups.len(),
                complete: false,
                emitted: false,
            });
            a.entry = Some(entry);
            a.ready = now + 1;
            if self.returned[sm] {
                self.next[sm] = 1;
            }
            self.audit[2] += 1;
            self.address[sm] = Some(a);
            return;
        }
        let line = *a.groups.front().unwrap();
        let r = Request {
            token: a.token,
            entry: a.entry.unwrap(),
            line,
            write: ops[a.token.op].kind == STORE,
        };
        let accepted = if a.shared {
            if self.lds_busy[sm] > now {
                false
            } else {
                let mut counts = vec![0usize; h[LDB]];
                for address in &ops[a.token.op].addresses {
                    if address / h[LINE] as u64 * h[LINE] as u64 == line {
                        counts[(address / 4) as usize % h[LDB]] += 1;
                    }
                }
                let rounds = counts.into_iter().max().unwrap().div_ceil(h[LDP]) as u64;
                self.lds_busy[sm] = now + rounds + u64::from(r.write);
                self.lds[sm].push_back((now + rounds + 2, r));
                b.stats[LDS_ROUNDS] += rounds;
                true
            }
        } else {
            let c = &mut self.l1[sm];
            let matching = c.primary.iter().flatten().find(|p| p.line == line);
            let capacity = matching
                .map(|p| p.targets.len() < h[L1SUB])
                .unwrap_or_else(|| c.primary.iter().any(Option::is_none));
            if c.core.is_none()
                && c.lower.is_empty()
                && !c.primary.iter().flatten().any(|p| p.releasing)
                && capacity
            {
                c.core = Some(r);
                true
            } else {
                false
            }
        };
        if accepted {
            self.audit[0] += 1;
            a.groups.pop_front();
        }
        if !a.groups.is_empty() {
            self.address[sm] = Some(a);
        }
    }
}
impl Engine<'_> {
    pub fn address_causal(&mut self) {
        let mut m = self.causal.take().unwrap();
        for sm in 0..self.b.hw[SMS] {
            if m.address[sm].is_none() {
                if let Some(t) = self.lsu_buffer[sm].filter(|t| t.time < self.clock) {
                    let op = &self.ops[t.op];
                    let mut groups = VecDeque::new();
                    for address in &op.addresses {
                        let line = address / self.b.hw[LINE] as u64 * self.b.hw[LINE] as u64;
                        if !groups.contains(&line) {
                            groups.push_back(line);
                        }
                    }
                    let shared = op
                        .addresses
                        .iter()
                        .all(|a| (0x70000000..0x70000000 + self.b.hw[LDS] as u64).contains(a));
                    assert!(
                        op.addresses
                            .iter()
                            .all(|a| ((0x70000000..0x70000000 + self.b.hw[LDS] as u64)
                                .contains(a))
                                == shared),
                        "mixed LDS/global instruction requires a separate address-space model"
                    );
                    m.address[sm] = Some(Address {
                        token: t,
                        ready: self.clock + 2,
                        groups,
                        entry: None,
                        shared,
                    });
                    self.lsu_buffer[sm] = None;
                }
            }
            m.admit(sm, self.clock, &mut self.b, &self.ops);
        }
        self.causal = Some(m);
    }
    pub fn step_memory(&mut self) {
        if self.waiting.is_empty() && self.sm_blocks.iter().all(|n| *n == 0) {
            self.causal.as_mut().unwrap().flush(self.clock, &mut self.b);
        }
        for q in &mut self.lsu_output {
            q.clear();
        }
        let free: Vec<_> = (0..self.b.hw[SMS])
            .map(|sm| {
                let tag = self.causal.as_ref().unwrap().used[sm]
                    .iter()
                    .flatten()
                    .find(|i| i.complete)
                    .map(|i| i.token);
                let Some(t) = tag else {
                    return true;
                };
                if self.ops[t.op].kind == STORE {
                    return true;
                }
                let path = self.ops[t.op].dest.unwrap_or(0) / 256;
                let mut alu = false;
                let mut fpu = false;
                for p in &self.sm_pipes[sm] {
                    let p = &self.pipes[*p];
                    if [FADD, FMA].contains(&p.kind) {
                        continue;
                    }
                    if let Some(t) = p.output() {
                        let o = &self.ops[t.op];
                        if o.dest.unwrap_or(0) / 256 == path {
                            if [SCALAR, VECTOR].contains(&o.kind) {
                                alu = true;
                            } else if Self::fpu(o.kind) {
                                fpu = true;
                            }
                        }
                    }
                }
                (alu as usize + fpu as usize) < self.b.hw[WB]
            })
            .collect();
        let outputs = self
            .causal
            .as_mut()
            .unwrap()
            .step(self.clock, &mut self.b, &free);
        for t in outputs {
            if self.ops[t.op].kind == STORE {
                self.retire(t, self.clock);
            } else {
                self.lsu_output[t.sm].push_back(t);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wrapper_has_five_nonpipe_slots_and_registered_delay() {
        let mut e = External::new(1);
        for id in 0..5 {
            e.insert(id, id as u64, 0, 2, 1);
        }
        assert_eq!(e.response(2), None);
        assert_eq!(e.response(3), Some(0));
        assert_eq!(e.ready_slot(0x90000000, 128, 3), None);
        e.slots[0] = None;
        assert_eq!(e.ready_slot(0x90000000, 128, 4), Some(0));
    }
    #[test]
    fn ready_priority_can_change_while_response_is_blocked() {
        let mut e = External::new(1);
        e.insert(0, 10, 0, 4, 1);
        e.insert(1, 11, 0, 0, 1);
        assert_eq!(e.response(1), Some(1));
        assert_eq!(e.response(5), Some(0));
        assert_eq!(e.slots.iter().flatten().count(), 2);
    }
}
