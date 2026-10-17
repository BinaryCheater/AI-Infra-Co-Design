//! Native recurrence of the explicitly opt-in Python source-frontend-v3 model.
//! This preserves its documented hit-cache and transaction-calendar approximations.
use super::*;
#[derive(Clone)]
struct Miss {
    line: u64,
    targets: usize,
    returning: Time,
    release: Time,
}
pub(super) struct Acceptance {
    pending: Vec<Vec<Miss>>,
    pub cursor: Vec<Option<Time>>,
}
impl Acceptance {
    pub fn new(sms: usize) -> Self {
        Self {
            pending: vec![Vec::new(); sms],
            cursor: vec![None; sms],
        }
    }
}
impl Builder<'_> {
    pub fn global_frontend(&mut self, ready: Time, sm: usize, line: u64, write: bool) -> Time {
        let mut a = self.acceptance.take().unwrap();
        let mut gate = ready.max(a.cursor[sm].map(|t| t + 1).unwrap_or(0));
        let same = loop {
            let active: Vec<usize> = a.pending[sm]
                .iter()
                .enumerate()
                .filter(|(_, p)| p.release > gate)
                .map(|(i, _)| i)
                .collect();
            let same = active
                .iter()
                .copied()
                .find(|i| a.pending[sm][*i].line == line);
            let mut next = gate;
            for i in &active {
                let p = &a.pending[sm][*i];
                if p.returning <= gate {
                    next = next.max(p.release);
                }
            }
            if let Some(i) = same {
                if a.pending[sm][i].targets >= self.hw[L1SUB] {
                    next = next.max(a.pending[sm][i].release);
                }
            } else if active.len() >= self.hw[L1M] {
                next = next.max(
                    active
                        .iter()
                        .map(|i| a.pending[sm][*i].release)
                        .min()
                        .unwrap(),
                );
            }
            if next == gate {
                break same;
            }
            gate = next;
        };
        a.cursor[sm] = Some(gate);
        let before = self.stats[L1_READ_MISS];
        let mut done = self.global(gate, sm, line, write);
        if self.stats[L1_READ_MISS] > before {
            let returning = done - 3;
            a.pending[sm].push(Miss {
                line,
                targets: 1,
                returning,
                release: returning + 2,
            });
        } else if let Some(i) = same {
            let p = &mut a.pending[sm][i];
            if gate < p.returning {
                p.targets += 1;
                p.release = p.returning + p.targets as u64 + 1;
                done = done.max(p.returning + p.targets as u64 + 2);
            }
        }
        self.acceptance = Some(a);
        done
    }
}
#[derive(Clone)]
struct ReadState {
    token: Token,
    slots: Vec<usize>,
    regs: Vec<usize>,
    grants: Vec<Option<Time>>,
    scheduled: bool,
}
pub(super) struct Frontend {
    slots: Vec<Vec<usize>>,
    packets: Vec<VecDeque<Vec<usize>>>,
    unpack: Vec<VecDeque<usize>>,
    fetch_index: Vec<usize>,
    fetch_pipe: Vec<(Time, usize, usize)>,
    rf: Vec<Vec<Option<ReadState>>>,
    pivots: Vec<Vec<Vec<usize>>>,
    pub reuse: Vec<Vec<Time>>,
    states: Vec<u8>,
    next: Vec<u8>,
    used: Vec<Vec<Option<usize>>>,
    complete: Vec<Vec<Option<Token>>>,
    tags: Vec<Vec<Option<Token>>>,
    pub arrivals: Vec<VecDeque<(usize, Token)>>,
    accepted: Vec<bool>,
    allocated: Vec<Option<Time>>,
    address: Vec<Option<(Token, Time)>>,
    pub max_cycles: Time,
    counters: [u64; 8],
}
impl Engine<'_> {
    pub fn enable_frontend(&mut self, pcs: Vec<u64>, slots: Vec<Vec<usize>>, max_cycles: Time) {
        for s in &self.streams {
            assert!(pcs[s.ops[0]] % 8 == 0, "frontend initial PC alignment");
            for (i, id) in s.ops.iter().enumerate() {
                assert!(pcs[*id] < (1u64 << 32) && pcs[*id] % 4 == 0, "frontend PC");
                if i > 0 {
                    assert_eq!(pcs[*id], pcs[s.ops[i - 1]] + 4, "frontend contiguous PC");
                }
                if self.ops[*id].kind == ENDPRG {
                    assert_eq!(i + 1, s.ops.len(), "ENDPRG must terminate warp");
                    assert!(self.ops[*id].sources.is_empty() && self.ops[*id].dest.is_none());
                }
            }
        }
        let h = self.b.hw;
        let ns = self.streams.len();
        self.trace = vec![[u64::MAX; 5]; self.ops.len()];
        self.frontend = Some(Frontend {
            slots,
            packets: vec![VecDeque::new(); ns],
            unpack: vec![VecDeque::new(); ns],
            fetch_index: vec![0; ns],
            fetch_pipe: Vec::new(),
            rf: vec![vec![None; h[COL]]; h[SMS]],
            pivots: vec![vec![vec![1; h[RFB]]; 2]; h[SMS]],
            reuse: vec![vec![0; h[COL]]; h[SMS]],
            states: vec![0; h[SMS]],
            next: vec![0; h[SMS]],
            used: vec![vec![None; h[LSU]]; h[SMS]],
            complete: vec![vec![None; h[LSU]]; h[SMS]],
            tags: vec![vec![None; h[LSU]]; h[SMS]],
            arrivals: vec![VecDeque::new(); h[SMS]],
            accepted: vec![false; h[SMS]],
            allocated: vec![None; h[SMS]],
            address: vec![None; h[SMS]],
            max_cycles,
            counters: [0; 8],
        });
    }
    pub fn export_frontend(&self) -> Vec<u64> {
        let f = self.frontend.as_ref().unwrap();
        let first = self.trace.iter().map(|v| v[0]).min().unwrap();
        let last = self
            .trace
            .iter()
            .filter_map(|v| if v[3] != u64::MAX { Some(v[3]) } else { None })
            .max()
            .unwrap_or(u64::MAX);
        let mut out = vec![first, last];
        out.extend(f.counters);
        for row in &self.trace {
            out.extend(row);
        }
        out
    }
    pub fn collect_frontend(&mut self) {
        let mut f = self.frontend.take().unwrap();
        let h = self.b.hw;
        let mut choices = vec![None::<(usize, usize)>; h[SMS]];
        let mut can_unpack = vec![false; self.streams.len()];
        for st in 0..self.streams.len() {
            let s = &self.streams[st];
            if let Some(sm) = self.blocks[s.block].sm {
                let wid = self.blocks[s.block].wids[s.warp];
                if f.packets[st].len() < 2
                    && f.fetch_index[st] < s.ops.len()
                    && choices[sm].is_none_or(|(w, _)| wid < w)
                {
                    choices[sm] = Some((wid, st));
                }
                can_unpack[st] = self.ready(st).is_some();
            }
        }
        for sm in 0..h[SMS] {
            if self.sm_blocks[sm] == 0 {
                continue;
            }
            let initial: Vec<usize> = (0..h[COL])
                .filter(|c| self.collectors[sm][*c].is_none() && f.reuse[sm][*c] <= self.clock)
                .collect();
            for path in 0..2 {
                let Some(free) = (0..h[COL])
                    .find(|c| self.collectors[sm][*c].is_none() && f.reuse[sm][*c] <= self.clock)
                else {
                    continue;
                };
                let demux = h[COL] == h[WARPS] && h[WARPS] > 1;
                if path == 1 && demux && initial.len() < 2 {
                    continue;
                }
                let mut candidate = None::<(usize, usize, Time, usize)>;
                for index in 0..self.resident_streams[sm].len() {
                    let st = self.resident_streams[sm][index];
                    let s = &self.streams[st];
                    if s.pc == s.ops.len()
                        || f.unpack[st].front().copied() != Some(s.ops[s.pc])
                        || self.path(s.ops[s.pc]) != path
                    {
                        continue;
                    }
                    if let Some(ready) = self.ready(st) {
                        let s = &self.streams[st];
                        let wid = self.blocks[s.block].wids[s.warp];
                        let rank = (wid + h[WARPS] - self.rr_collect[sm][path]) % h[WARPS];
                        if candidate.is_none_or(|x| rank < x.0) {
                            candidate = Some((rank, st, ready, wid));
                        }
                    }
                }
                let Some((_, st, ready, wid)) = candidate else {
                    continue;
                };
                let collector = if path == 1 && demux { initial[1] } else { free };
                let id = self.streams[st].ops[self.streams[st].pc];
                let op = &self.ops[id];
                let accept = ready.max(self.clock);
                let token = Token {
                    op: id,
                    sm,
                    wid,
                    time: accept,
                };
                self.trace[id][0] = accept;
                self.collectors[sm][collector] = Some(token);
                self.streams[st].pc += 1;
                self.streams[st].busy = true;
                for dst in op.dest.iter() {
                    if *dst != 256 {
                        self.streams[st].regs[*dst] = id;
                    }
                }
                if op.kind < 21 {
                    self.b.stats[5 + op.kind] += 1;
                }
                if op.sources.is_empty() {
                    self.trace[id][1] = accept + 1;
                    self.schedule(
                        Token {
                            time: accept + 1,
                            ..token
                        },
                        0,
                        collector,
                    );
                } else {
                    f.rf[sm][collector] = Some(ReadState {
                        token,
                        slots: f.slots[id].clone(),
                        regs: op.sources.clone(),
                        grants: vec![None; op.sources.len()],
                        scheduled: false,
                    });
                }
                self.rr_collect[sm][path] = (wid + 1) % h[WARPS];
                f.unpack[st].pop_front();
            }
        }
        for sm in 0..h[SMS] {
            let mut requests = vec![vec![Vec::<(usize, usize, usize)>::new(); h[RFB]]; 2];
            for c in 0..h[COL] {
                if let Some(r) = &f.rf[sm][c] {
                    for i in 0..r.regs.len() {
                        if r.grants[i].is_none_or(|g| self.clock < g + 2) {
                            let reg = r.regs[i];
                            requests[reg / 256][(r.token.wid + reg % 256) % h[RFB]].push((
                                c * 4 + r.slots[i],
                                c,
                                i,
                            ));
                        }
                    }
                    if r.grants
                        .iter()
                        .all(|g| g.is_some_and(|g| self.clock >= g + 2))
                    {
                        f.rf[sm][c] = None;
                    }
                }
            }
            for path in 0..2 {
                for bank in 0..h[RFB] {
                    let requests = &mut requests[path][bank];
                    for _ in 0..h[RFR].min(requests.len()) {
                        let index = (0..requests.len())
                            .min_by_key(|i| {
                                (requests[*i].0 + 4 * h[COL] - f.pivots[sm][path][bank])
                                    % (4 * h[COL])
                            })
                            .unwrap();
                        let (requester, c, i) = requests.remove(index);
                        f.pivots[sm][path][bank] = (requester + 1) % (4 * h[COL]);
                        let r = f.rf[sm][c].as_mut().unwrap();
                        if r.grants[i].is_none() {
                            r.grants[i] = Some(self.clock);
                        } else {
                            f.counters[2] += 1;
                        }
                    }
                }
            }
            for c in 0..h[COL] {
                if let Some(r) = f.rf[sm][c].as_mut() {
                    if !r.scheduled && r.grants.iter().all(Option::is_some) {
                        let when = r.grants.iter().map(|g| g.unwrap() + 3).max().unwrap();
                        self.trace[r.token.op][1] = when;
                        self.schedule(
                            Token {
                                time: when,
                                ..r.token
                            },
                            0,
                            c,
                        );
                        r.scheduled = true;
                    }
                }
            }
        }
        for st in 0..self.streams.len() {
            if can_unpack[st] && f.unpack[st].is_empty() {
                if let Some(packet) = f.packets[st].pop_front() {
                    f.unpack[st].extend(packet);
                }
            }
        }
        let responses = f.fetch_pipe.clone();
        for entry in responses {
            let (when, st, index) = entry;
            if when == self.clock {
                if let Some(pos) = f.fetch_pipe.iter().position(|e| *e == entry) {
                    f.fetch_pipe.remove(pos);
                } else {
                    continue;
                }
                if f.packets[st].len() < 2 {
                    f.packets[st].push_back(
                        self.streams[st].ops[index..(index + 2).min(self.streams[st].ops.len())]
                            .to_vec(),
                    );
                    f.counters[0] += 1;
                } else {
                    f.fetch_index[st] = index;
                    f.fetch_pipe.retain(|e| e.1 != st);
                    f.counters[1] += 1;
                }
            }
        }
        for (_, st) in choices.into_iter().flatten() {
            f.fetch_pipe.push((self.clock + 2, st, f.fetch_index[st]));
            f.fetch_index[st] += 2;
        }
        self.frontend = Some(f);
    }
    pub fn address_frontend(&mut self) {
        let mut f = self.frontend.take().unwrap();
        let h = self.b.hw;
        for sm in 0..h[SMS] {
            if f.address[sm].is_none()
                && self.lsu_buffer[sm].is_some()
                && self.address_until[sm] <= self.clock
            {
                let t = self.lsu_buffer[sm].take().unwrap();
                f.address[sm] = Some((t, t.time.max(self.clock)));
            }
            let Some((token, first)) = f.address[sm] else {
                continue;
            };
            if first + 2 > self.clock
                || f.states[sm] != 0
                || f.allocated[sm] == Some(self.clock)
                || f.used[sm].iter().all(Option::is_some)
            {
                continue;
            }
            let entry = f.used[sm].iter().position(Option::is_none).unwrap();
            f.used[sm][entry] = Some(token.op);
            f.tags[sm][entry] = Some(token);
            f.allocated[sm] = Some(self.clock);
            if f.accepted[sm] {
                f.next[sm] = 1;
                f.counters[7] += 1;
            }
            let done = self.b.memory(
                first.max(self.clock - 2),
                &self.ops[token.op],
                sm,
                token.wid,
            );
            self.address_until[sm] = self.b.last(Key(9, sm, 0, 0));
            f.address[sm] = None;
            self.schedule(
                Token {
                    time: done - 1,
                    ..token
                },
                3,
                entry,
            );
        }
        self.frontend = Some(f);
    }
    pub fn writebacks_frontend(&mut self) -> Vec<bool> {
        let mut f = self.frontend.take().unwrap();
        let h = self.b.hw;
        f.states = f.next.clone();
        f.accepted.fill(false);
        let before: Vec<Vec<bool>> = f
            .complete
            .iter()
            .map(|r| r.iter().map(Option::is_some).collect())
            .collect();
        let free: Vec<usize> = f
            .used
            .iter()
            .map(|r| r.iter().position(Option::is_none).unwrap_or(0))
            .collect();
        let mut ready = vec![true; h[SMS]];
        for sm in 0..h[SMS] {
            let entry = f.complete[sm].iter().position(Option::is_some).unwrap_or(0);
            if let Some(tag) = f.tags[sm][entry] {
                if self.ops[tag.op].kind != STORE {
                    let path = self.ops[tag.op].dest.unwrap_or(0) / 256;
                    let mut alu = false;
                    let mut fpu = false;
                    for &p in &self.sm_pipes[sm] {
                        let pipe = &self.pipes[p];
                        if [FADD, FMA].contains(&pipe.kind) {
                            continue;
                        }
                        if let Some(t) = pipe.output() {
                            let op = &self.ops[t.op];
                            if op.dest.unwrap_or(0) / 256 == path {
                                if [SCALAR, VECTOR].contains(&op.kind) {
                                    alu = true;
                                } else if [FADD, FMA, FMAX, FTOI, ITOF].contains(&op.kind) {
                                    fpu = true;
                                }
                            }
                        }
                    }
                    ready[sm] = (alu as usize + fpu as usize) < h[WB];
                }
            }
            self.lsu_output[sm].clear();
            if f.states[sm] == 2 {
                if let Some(t) = f.complete[sm].iter().flatten().next() {
                    self.lsu_output[sm].push_back(*t);
                }
            }
            if f.states[sm] == 0 {
                if let Some((entry, t)) = f.arrivals[sm].pop_front() {
                    f.complete[sm][entry] = Some(t);
                    f.accepted[sm] = true;
                    f.counters[6] += 1;
                }
            }
            if self.lsu_output[sm]
                .front()
                .is_some_and(|t| self.ops[t.op].kind == STORE)
            {
                let t = self.lsu_output[sm].pop_front().unwrap();
                self.retire(t, t.time.max(self.clock));
            }
        }
        let consumed = self.writebacks();
        for sm in 0..h[SMS] {
            for i in 0..h[LSU] {
                if f.used[sm][i].is_some_and(|op| self.completion[op].is_some()) {
                    f.used[sm][i] = None;
                    f.complete[sm][i] = None;
                }
            }
            f.next[sm] = match f.states[sm] {
                2 => {
                    let fired = f.complete[sm].iter().filter(|v| v.is_some()).count()
                        < before[sm].iter().filter(|v| **v).count();
                    if fired
                        && before[sm]
                            .iter()
                            .enumerate()
                            .any(|(i, v)| *v && i != free[sm])
                    {
                        2
                    } else {
                        0
                    }
                }
                1 => {
                    if ready[sm] && before[sm].iter().any(|v| *v) {
                        2
                    } else {
                        0
                    }
                }
                _ => {
                    if ready[sm] && f.complete[sm].iter().any(Option::is_some) {
                        2
                    } else {
                        0
                    }
                }
            };
            f.counters[3 + f.states[sm] as usize] += 1;
        }
        self.frontend = Some(f);
        consumed
    }
}

/// Derive the same saturated LDS hardware-control template as coalescer.py.
/// This does not observe a workload or substitute a measured instruction latency.
fn derive_shared_intervals(entries: usize) -> Vec<u64> {
    assert!(entries > 0);
    let (mut address, mut state) = (0u8, 0u8);
    let mut used = vec![false; entries];
    let mut complete = vec![false; entries];
    let mut request_tag = None;
    let mut responses = VecDeque::<(u64, usize)>::new();
    let mut outputs = Vec::new();
    for cycle in 0..2048 {
        let free = used.iter().position(|v| !v).unwrap_or(0);
        let output = complete.iter().position(|v| *v).unwrap_or(0);
        let allocate = address == 1 && state == 0 && used.iter().any(|v| !v);
        let response = state == 0 && responses.front().is_some_and(|(t, _)| *t <= cycle);
        let fire = state == 2 && complete.iter().any(|v| *v);
        let next = match state {
            0 if response => {
                if allocate {
                    1
                } else {
                    2
                }
            }
            0 if complete.iter().any(|v| *v) => 2,
            0 => 0,
            // Preserve the source valid_entry next-state test, including its bubble.
            2 => {
                if complete.iter().enumerate().any(|(i, v)| *v && i != free) {
                    2
                } else {
                    0
                }
            }
            1 => {
                if complete.iter().any(|v| *v) {
                    2
                } else {
                    0
                }
            }
            _ => unreachable!(),
        };
        if state == 0 {
            if response {
                let (_, tag) = responses.pop_front().unwrap();
                complete[tag] = true;
                if allocate {
                    request_tag = Some(free);
                }
            } else if allocate {
                used[free] = true;
                request_tag = Some(free);
            }
        } else if state == 1 {
            used[free] = true;
            request_tag = Some(free);
        } else if fire {
            used[output] = false;
            complete[output] = false;
            outputs.push(cycle);
        }
        if address == 0 {
            address = 1;
        } else if address == 1 && allocate {
            address = 2;
        } else if address == 2 {
            responses.push_back((cycle + 3, request_tag.unwrap()));
            address = 0;
        }
        state = next;
    }
    let intervals: Vec<u64> = outputs.windows(2).map(|v| v[1] - v[0]).collect();
    assert!(
        intervals.len() >= 64,
        "no bounded shared coalescer service template found"
    );
    let tail = &intervals[intervals.len() - 64..];
    for period in 1..=16 {
        if tail
            .iter()
            .enumerate()
            .all(|(i, value)| *value == tail[i % period])
        {
            return tail[..period].to_vec();
        }
    }
    panic!("no bounded shared coalescer service template found");
}

pub(super) fn shared_intervals(entries: usize) -> Vec<u64> {
    assert!(entries > 0, "shared template needs positive entry count");
    use std::sync::{Mutex, OnceLock};
    static TEMPLATES: OnceLock<Mutex<HashMap<usize, Vec<u64>>>> = OnceLock::new();
    let mut templates = TEMPLATES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    templates
        .entry(entries)
        .or_insert_with(|| derive_shared_intervals(entries))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saturated_template_matches_source_reference_geometries() {
        // Independent outputs from the pinned Python control recurrence.
        for (entries, expected) in [
            (1, vec![6]),
            (2, vec![4, 3]),
            (4, vec![4]),
            (8, vec![4]),
            (16, vec![4]),
            (32, vec![4]),
        ] {
            assert_eq!(derive_shared_intervals(entries), expected);
            assert_eq!(shared_intervals(entries), expected);
            assert_eq!(shared_intervals(entries), expected);
        }
    }
    #[test]
    fn all_entry_counts_through_32_match_independent_reference() {
        for entries in 1..=32 {
            let expected = match entries {
                1 => vec![6],
                2 => vec![4, 3],
                _ => vec![4],
            };
            assert_eq!(shared_intervals(entries), expected);
        }
    }
    #[test]
    fn zero_entries_reject_without_poisoning_template_cache() {
        assert!(std::panic::catch_unwind(|| shared_intervals(0)).is_err());
        assert_eq!(shared_intervals(8), vec![4]);
    }
    #[test]
    fn frontend_factory_rejects_64_lanes_before_mask_construction() {
        let mut hardware = [1u64; 32];
        hardware[3] = 64;
        unsafe {
            assert!(vt_frontend_create(hardware.as_ptr()).is_null());
            hardware[3] = 32;
            let session = vt_frontend_create(hardware.as_ptr());
            assert!(!session.is_null());
            vt_destroy(session);
        }
    }
}
