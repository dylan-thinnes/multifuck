use std::collections::BTreeMap;
use std::io;
use std::io::{BufReader, BufRead};
use std::fs::File;
use std::hash::{Hash};
use std::thread;
use std::fmt;

use std::path::PathBuf;
use structopt::StructOpt;

use std::sync;
use std::sync::mpsc;
use std::collections::VecDeque;

use cursive_aligned_view::Alignable;
use cursive::{
    Cursive,
    XY,
    theme::{
        Style,
        ColorStyle,
        ColorType,
        Color,
        BaseColor,
        PaletteColor::{Primary}
    },
    utils::span::SpannedString,
    view::{View, ViewWrapper, scroll::{ScrollStrategy}, Scrollable, Resizable},
    views::{ResizedView, LinearLayout, Panel, TextView, TextContent, EditView, NamedView, Button},
};

#[derive(Debug, StructOpt)]
#[structopt(name = "multifuck", version = "0.0.1")]
struct Opt {
    /// Brainfuck source to run, stdin if not present
    #[structopt(parse(from_os_str))]
    source: PathBuf,

    /// Inputs to give to program before reading stdin
    inputs: Vec<String>,

    /// Run program with graphical interface (uses ncurses)
    #[structopt(long, short)]
    gui: bool,

    /// Print diagnostic info (only relevant in CLI-mode)
    #[structopt(long, short)]
    verbose: bool,

    /// Output printed numbers in ASCII mode
    #[structopt(long, short)]
    ascii: bool
}

struct InputLog {
    history: Vec<ThreadIO>,
    queued: VecDeque<String>
}

impl InputLog {
    fn to_str (&self) -> SpannedString<Style> {
        let mut span = SpannedString::<Style>::plain("");

        for input in self.history.iter() {
            span.append_plain(input.debug_fmt());
            span.append_plain("\n");
        }

        for pre_input in self.queued.iter() {
            span.append_plain(ThreadIO::debug_fmt_unknown(&pre_input));
            span.append_plain("\n");
        }

        span
    }
}

struct Inputter {
    inputs: sync::Arc<sync::RwLock<InputLog>>,
    need_input_tx: mpsc::Sender<Option<ThreadIO>>,
}

impl Inputter {
    fn new (queued_inputs: Vec<String>) -> (Self, mpsc::Sender<String>, mpsc::Receiver<Option<ThreadIO>>) {
        let (need_input_tx, need_input_rx) = mpsc::channel::<Option<ThreadIO>>();
        let (give_input_tx, give_input_rx) = mpsc::channel::<String>();

        let inputs =
                sync::Arc::new(sync::RwLock::new(InputLog {
                    history: vec![],
                    queued: VecDeque::from(queued_inputs)
                }));
        let inputs_c = inputs.clone();

        thread::spawn(move || {
            while let Ok(s) = give_input_rx.recv() {
                match inputs_c.write() {
                    Ok(mut guard) => guard.queued.push_back(s.to_string()),
                    Err(_) => break
                }
            }
        });

        let inputter = Inputter { inputs, need_input_tx };
        (inputter, give_input_tx, need_input_rx)
    }

    fn to_input_log (&self) -> SpannedString<Style> {
        self.inputs.read().expect("Cannot get lock on input queue!").to_str()
    }

    fn recv (&mut self, thread: &Thread) -> String {
        let mut first_loop = true;

        // spin until an input is poppable
        let output = loop {
            match self.inputs.write() {
                Ok(mut guard) => {
                    match guard.queued.pop_front() {
                        Some(s) => {
                            guard.history.push(thread.io(s.clone()));
                            break s;
                        },
                        None => {
                            if first_loop {
                                let signal = ThreadIO::from(thread.io(String::from("")));
                                self.need_input_tx.send(Some(signal));
                            }
                        }
                    }
                }
                Err(_) => panic!("Cannot get lock on input queue!")
            }

            first_loop = false;
        };

        self.need_input_tx.send(None);
        output
    }
}

fn main () -> io::Result<()> {
    let opt = Opt::from_args();

    let f = File::open(opt.source)?;
    let input_program = Box::new(BufReader::new(f));

    let (tape, source_map) = Tape::parse(input_program)?;
    let mut state = State {
        tape, source_map,
        memory: Memory::new(),
        threads: vec![
            Thread::new()
        ],
        outputs: vec![],
        global_cycle_count: 0
    };

    let (mut inputter, give_input_tx, need_input_rx) = Inputter::new(opt.inputs);
    let input_log_c = inputter.inputs.clone();

    if !opt.gui {
        thread::spawn(move || {
            loop {
                let mut raw_inp = String::new();
                io::stdin().read_line(&mut raw_inp);
                give_input_tx.send(raw_inp);
            }
        });

        if opt.verbose {
            thread::spawn(move || {
                loop {
                    match need_input_rx.recv() {
                        Err(_) => break,
                        Ok(None) => continue,
                        Ok(Some(thread_io)) =>
                            eprintln!("Input needed (#{}, ${}): ", thread_io.cycle, thread_io.thread_id)
                    }
                }
            });
        }

        let mut output_idx = 0;

        while state.step(opt.ascii, &mut inputter) {
            for thread_io in state.outputs[output_idx..].iter() {
                if !opt.verbose {
                    let output = thread_io.stdout_fmt(opt.ascii);
                    print!("{}", output);
                } else {
                    println!("{}", thread_io.debug_fmt());
                }
            }
            output_idx = state.outputs.len();
        }

        for thread in state.threads.iter() {
            eprintln!("${}: #{} + #{} = #{}", thread.id, thread.started_at, thread.steps_taken, thread.curr_cycle());
        }
    } else {
        let (tx, rx) = mpsc::channel::<UICommand>();

        let source_content = TextContent::new("");
        let step_size_view = TextContent::new("");
        let output_stream_content = TextContent::new("");
        let input_stream_content = TextContent::new("");
        let input_stream_content_c = input_stream_content.clone();
        let input_needed_content = TextContent::new("");
        let memory_content = TextContent::new("");

        let step_btn_sender = tx.clone();
        let decr_step_btn_sender = tx.clone();
        let incr_step_btn_sender = tx.clone();

        let mut siv = cursive::default();
        let cb_sink_send_input = siv.cb_sink().clone();

        siv.add_fullscreen_layer(
            LinearLayout::horizontal()
            .child(
                Panel::new(
                    LinearLayout::vertical()
                    .child(
                        ResizedView::with_full_height(
                            TextView::new_with_content(source_content.clone())
                        )
                        .scrollable()
                    )
                    .child(
                        LinearLayout::horizontal()
                        .child(NamedView::new("next_step",
                            Button::new("Next Step", move |_| { step_btn_sender.send(UICommand::Step); })
                        ))
                        .child(TextView::new(" | Step Size: "))
                        .child(Button::new("-", move |_| { decr_step_btn_sender.send(UICommand::DecrStepSize); }))
                        .child(TextView::new(" "))
                        .child(TextView::new_with_content(step_size_view.clone()))
                        .child(TextView::new(" "))
                        .child(Button::new("+", move |_| { incr_step_btn_sender.send(UICommand::IncrStepSize); }))
                        .align_bottom_center()
                        .fixed_height(1)
                    )
                )
                .title("Source")
                .percent_width((0.0, 0.5))
            )
            .child(
                LinearLayout::vertical()
                .child(
                    LinearLayout::horizontal()
                    .child(
                        Panel::new(
                            TextView::new_with_content(output_stream_content.clone())
                            .scrollable()
                            .scroll_strategy(ScrollStrategy::StickToBottom)
                        )
                        .title("Output")
                        .percent_width((0.0, 0.5))
                    )
                    .child(
                        Panel::new(
                            LinearLayout::vertical()
                            .child(
                                ResizedView::with_full_height(
                                    TextView::new_with_content(input_stream_content.clone())
                                )
                                .scrollable()
                                .scroll_strategy(ScrollStrategy::StickToBottom)
                            )
                            .child(
                                TextView::new_with_content(input_needed_content.clone())
                            )
                            .child(
                                NamedView::new("user_input_editor",
                                    EditView::new()
                                    .on_submit(move |cursive, raw_inp| {
                                        let editor = cursive.find_name::<EditView>("user_input_editor");
                                        if let Some(mut editor) = editor {
                                            editor.set_content("");
                                        }
                                        give_input_tx.send(raw_inp.to_string());

                                        let input_log = input_log_c.read().expect("Could not unlock input log!");
                                        input_stream_content_c.set_content(input_log.to_str());

                                        cb_sink_send_input.send(Box::new(Cursive::noop)).unwrap();
                                    })
                                )
                                .full_width()
                                .fixed_height(1)
                            )
                        )
                        .title("Input")
                        .percent_width((0.5, 1.0))
                    )
                    .percent_height((0.0, 0.5))
                )
                .child(
                    Panel::new(TextView::new_with_content(memory_content.clone()))
                    .title("Memory")
                    .percent_height((0.5, 1.0))
                )
                .percent_width((0.5, 1.0))
            )
        );

        let cb_sink = siv.cb_sink().clone();

        let exit_sender = tx.clone();
        siv.add_global_callback('q', move |s| {
            s.quit();
            if exit_sender.send(UICommand::Exit).is_err() {
                return;
            }
        });

        let step_sender = tx.clone();
        siv.add_global_callback(' ', move |_| {
            if step_sender.send(UICommand::Step).is_err() {
                return;
            }
            if step_sender.send(UICommand::Repaint).is_err() {
                return;
            }
        });

        let incr_step_sender = tx.clone();
        siv.add_global_callback('+', move |_| {
            if incr_step_sender.send(UICommand::IncrStepSize).is_err() {
                return;
            }
        });

        let decr_step_sender = tx.clone();
        siv.add_global_callback('-', move |_| {
            if decr_step_sender.send(UICommand::DecrStepSize).is_err() {
                return;
            }
        });


        let copied_ascii_mode: bool = opt.ascii;
        let handle = thread::spawn(move || {
            let mut step_size = 1;

            while let Ok(ui_command) = rx.recv() {
                match ui_command {
                    UICommand::Repaint => {
                    },
                    UICommand::Step => {
                        for _ in 0..step_size {
                            input_stream_content.set_content(inputter.to_input_log());
                            if !state.step(copied_ascii_mode, &mut inputter) {
                                break;
                            }
                        }
                    },
                    UICommand::IncrStepSize => {
                        step_size *= 2;
                    },
                    UICommand::DecrStepSize => {
                        if step_size > 1 {
                            step_size /= 2;
                        }
                    },
                    UICommand::Exit => {
                        break;
                    },
                }

                source_content.set_content(state.spanned_tape());
                output_stream_content.set_content(state.output_log());
                input_stream_content.set_content(inputter.to_input_log());
                memory_content.set_content(format!["{:?}", state.memory]);
                step_size_view.set_content(format!("{:#5}", step_size));
                cb_sink.send(Box::new(Cursive::noop)).unwrap();

                // clear any commands that may have queued in the meantime
                loop {
                    if let Err(_) = rx.try_recv() { break; }
                }
            }

            source_content.set_content(state.spanned_tape());
            output_stream_content.set_content(state.output_log());
            input_stream_content.set_content(inputter.to_input_log());
            memory_content.set_content(format!["{:?}", state.memory]);
            step_size_view.set_content(format!("{:#5}", step_size));
            cb_sink.send(Box::new(Cursive::noop)).unwrap();
        });

        let cb_sink_need_input = siv.cb_sink().clone();
        thread::spawn(move || {
            loop {
                match need_input_rx.recv().unwrap() {
                    None => {
                        input_needed_content.set_content("");
                        cb_sink_need_input.send(Box::new(|siv| {
                            siv.focus_name("next_step");
                        }));
                    },
                    Some(thread_io) => {
                        let text = format!("#{}, ${} needs input:", thread_io.cycle, thread_io.thread_id);
                        let spanned = SpannedString::styled(text, ColorStyle::highlight());
                        input_needed_content.set_content(spanned);
                        cb_sink_need_input.send(Box::new(|siv| {
                            siv.focus_name("user_input_editor");
                        }));
                    }
                }

                cb_sink_need_input.send(Box::new(Cursive::noop)).unwrap();
            }
        });

        tx.send(UICommand::Repaint);
        siv.run();

        handle.join();
    }

    Ok(())
}

enum UICommand {
    Exit,
    Repaint,
    Step,
    DecrStepSize,
    IncrStepSize
}

#[derive(Debug)]
struct State {
    tape: Tape,
    memory: Memory,
    threads: Vec<Thread>,
    source_map: SourceMap,
    outputs: Vec<ThreadIO>,
    global_cycle_count: usize
}

#[derive(Debug)]
struct ThreadIO {
    thread_id: usize,
    cycle: usize,
    msg: String
}

impl ThreadIO {
    pub fn debug_fmt (&self) -> String {
        format!("#{}, ${}: {}", self.cycle, self.thread_id, self.msg)
    }

    pub fn stdout_fmt (&self, ascii_mode: bool) -> String {
        self.msg.clone() + if ascii_mode { "" } else { "\n" }
    }

    pub fn debug_fmt_unknown (msg: &String) -> String {
        format!("#{}, ${}: {}", "_", "_", msg)
    }
}

impl State {
    fn step (&mut self, ascii_mode: bool, inputter: &mut Inputter) -> bool {
        let mut at_least_one_live_thread = false;

        let mut memory_edits = vec![];
        let mut forks = vec![];

        for thread in &mut self.threads.iter_mut() {
            match thread.step(&mut self.tape, &mut self.memory, ascii_mode, inputter) {
                None => {},
                Some((maybe_output, memory_edit, fork)) => {
                    at_least_one_live_thread = true;

                    memory_edits.push(memory_edit);

                    if let Some(output) = maybe_output {
                        self.outputs.push(thread.io(output));
                    }

                    if fork { forks.push(thread.id); }
                }
            }
        }

        for thread_id in forks {
            let mut new_thread = self.threads[thread_id].clone();
            new_thread.id = self.threads.len();
            new_thread.started_at = self.global_cycle_count + 1;
            new_thread.sleep();
            self.threads.push(new_thread);
        }

        for memory_edit in memory_edits {
            self.memory.edit(memory_edit);
        }

        if at_least_one_live_thread {
            self.global_cycle_count += 1;
        }

        at_least_one_live_thread
    }

    fn output_log (&self) -> SpannedString<Style> {
        let mut span = SpannedString::<Style>::plain("");

        for output in &self.outputs {
            span.append_plain(output.debug_fmt());
            span.append_plain("\n");
        }

        span
    }

    fn spanned_tape (&self) -> SpannedString<Style> {
        let mut output = SpannedString::<Style>::plain("");

        let mut thread_locations: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for thread in self.threads.iter() {
            let instr_ptr_w_fork_offset =
                usize::saturating_sub(thread.instr_ptr, if thread.sleeping { 1 } else { 0 });

            match self.source_map.map.get(instr_ptr_w_fork_offset) {
                None => {}
                Some(span_index) => {
                    thread_locations.insert(*span_index, thread.id);
                }
            }
        }

        for (line_no, (line, spans)) in self.source_map.source.iter().enumerate() {
            let mut last_col = 0;

            for (span_no, span) in spans.iter().enumerate() {
                if span.start_col > last_col {
                    output.append_styled(
                        &line[last_col..span.start_col],
                        ColorStyle::secondary()
                    );
                }

                let color = match thread_locations.get(&(line_no, span_no)) {
                    None => ColorStyle::primary(),
                    Some(thread_id) => {
                        let base_color = BaseColor::from(1 + (*thread_id as u8 % 6));

                        ColorStyle {
                            front: ColorType::from(Primary),
                            back: ColorType::from(Color::Dark(base_color))
                        }
                    }
                };

                output.append_styled(&line[span.start_col..span.end_col], color);
                last_col = span.end_col;
            }

            output.append_styled(
                &line[last_col..],
                ColorStyle::secondary()
            );

            output.append_plain("\n");
        }

        output
    }
}

#[derive(Debug)]
struct Tape(Vec<MultiInstr>);

#[derive(Debug)]
struct SourceMap {
    source: Vec<(String,Vec<LexemeSpan>)>,
    map: Vec<(usize,usize)>
}

#[derive(Debug)]
struct LexemeSpan {
    start_col: usize,
    end_col: usize
}

impl Tape {
    fn get_instr (&self, idx: usize) -> Option<&MultiInstr> {
        self.0.get(idx)
    }

    fn parse<B: BufRead> (buff: B) -> io::Result<(Self, SourceMap)> {
        let mut internal = vec![];
        let mut source = vec![];
        let mut map = vec![];

        for (line_no, res_line) in buff.lines().enumerate() {
            let line = res_line?;
            let mut char_stream = line.chars().enumerate().peekable();
            let mut lexeme_bounds = vec![];

            while let Some((col_no, c)) = char_stream.next() {
                if c == '#' { break; }
                if let Some(instr) = Instr::parse(c) {
                    let start_col = col_no;
                    let mut repetition = None;
                    let mut end_col = start_col + 1;

                    if let Some((_, peeked_char)) = char_stream.peek() {
                        if *peeked_char == '{' {
                            char_stream.next();

                            let mut total_value = 0;
                            end_col += 1;
                            while let Some((_, num_c @ '0'..='9')) = char_stream.peek() {
                                end_col += 1;
                                total_value *= 10;
                                total_value += num_c.to_digit(10).unwrap();
                                char_stream.next();
                            }
                            end_col += 1;

                            repetition = Some(total_value as u16);

                            char_stream.next();
                        }
                    }

                    lexeme_bounds.push(LexemeSpan { start_col, end_col });
                    map.push((line_no, lexeme_bounds.len() - 1));
                    internal.push(MultiInstr { instr, repetition });
                }
            }

            source.push((line.clone(), lexeme_bounds));
        }

        Ok((Tape(internal), SourceMap { source, map }))
    }
}

#[derive(Debug, Clone)]
struct MultiInstr {
    instr: Instr,
    repetition: Option<u16>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Instr {
    Increment,
    Decrement,
    MoveForward,
    MoveBackward,
    StartLoop,
    EndLoop,
    Output,
    Input,
    Fork
}

impl Instr {
    fn parse (c: char) -> Option<Self> {
        match c {
            '+' => Some(Instr::Increment),
            '-' => Some(Instr::Decrement),
            '>' => Some(Instr::MoveForward),
            '<' => Some(Instr::MoveBackward),
            '[' => Some(Instr::StartLoop),
            ']' => Some(Instr::EndLoop),
            '.' => Some(Instr::Output),
            ',' => Some(Instr::Input),
            '&' => Some(Instr::Fork),
            _   => None
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Address(BTreeMap<Direction, i32>);
type Direction = i32;

struct Memory(BTreeMap<Address, i32>);

impl Memory {
    fn new () -> Self {
        Memory(BTreeMap::new())
    }

    fn edit (&mut self, edit: MemoryEdit) -> () {
        match edit {
            MemoryEdit::Delta(addr, x) =>
                mmod_map(&mut self.0, 0, false, addr, |y| y + x),
            MemoryEdit::Absolute(addr, x) =>
                mmod_map(&mut self.0, 0, false, addr, |_| x),
            MemoryEdit::NoEdit =>
                {}
        };
    }

    fn get (&self, addr: &Address) -> i32 {
        mget_def(&self.0, 0, addr.clone())
    }
}

impl fmt::Debug for Memory {
    fn fmt (&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(fmt)
    }
}

impl Address {
    fn new () -> Address { Address(BTreeMap::new()) }
}

impl fmt::Debug for Address {
    fn fmt (&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "(")?;

        match self.0.keys().next() {
            None => {}
            Some(k) => {
                let k = *k;
                for dim in k..0 {
                    if dim != k {
                        write!(fmt, ",")?;
                    }

                    let dist = *self.0.get(&dim).unwrap_or(&0);
                    write!(fmt, "{}", dist)?;
                }
            }
        }

        write!(fmt, "|")?;

        match self.0.keys().rev().next() {
            None => {}
            Some(k) => {
                let k = *k;
                for dim in 0..=k {
                    if dim != 0 {
                        write!(fmt, ",")?;
                    }

                    let dist = *self.0.get(&dim).unwrap_or(&0);
                    write!(fmt, "{}", dist)?;
                }
            }
        }

        write!(fmt, ")")
    }
}

#[derive(Debug, Clone)]
struct Thread {
    id: usize,
    instr_ptr: usize,
    memory_ptr: Address,
    direction: Direction,
    sleeping: bool,
    started_at: usize,
    steps_taken: usize
}

impl Thread {
    fn new () -> Thread {
        Thread {
            id: 0,
            instr_ptr: 0,
            memory_ptr: Address::new(),
            direction: 0,
            sleeping: false,
            started_at: 0,
            steps_taken: 0
        }
    }

    fn incr_instr (&mut self) { self.instr_ptr += 1; }
    fn decr_instr (&mut self) { self.instr_ptr -= 1; }
    fn curr_instr (&self, tape: &Tape) -> Option<MultiInstr> {
        tape.get_instr(self.instr_ptr).cloned()
    }

    fn sleep (&mut self) {
        self.sleeping = true;
    }

    fn io (&self, msg: String) -> ThreadIO {
        ThreadIO {
            thread_id: self.id,
            cycle: self.curr_cycle() + 1,
            msg
        }
    }

    fn step (&mut self, tape: &Tape, curr_memory: &Memory, ascii_mode: bool, inputter: &mut Inputter) -> Option<(Option<String>, MemoryEdit, bool)> {
        if self.sleeping {
            self.sleeping = false;
            return Some((None, MemoryEdit::NoEdit, false));
        }

        let multi_instr = self.curr_instr(tape)?;
        let instr = multi_instr.instr;

        match instr {
            Instr::MoveForward => {
                mmod_map(&mut self.memory_ptr.0, 0, true, self.direction, |c| c + 1);
                self.incr_instr();
            },
            Instr::MoveBackward => {
                mmod_map(&mut self.memory_ptr.0, 0, true, self.direction, |c| c - 1);
                self.incr_instr();
            },
            Instr::StartLoop => {
                let curr_val = curr_memory.get(&self.memory_ptr);
                if curr_val == 0 {
                    let mut depth = 0;

                    loop {
                        self.incr_instr();
                        let curr_multi_instr = self.curr_instr(tape)?;
                        let curr_instr = curr_multi_instr.instr;

                        if curr_instr == Instr::EndLoop && depth == 0 { break; }

                        if curr_instr == Instr::StartLoop {
                            depth += 1;
                        } else if curr_instr == Instr::EndLoop {
                            depth -= 1;
                        }
                    }
                }

                self.incr_instr();
            },
            Instr::EndLoop => {
                let mut depth = 0;

                loop {
                    self.decr_instr();
                    let curr_multi_instr = self.curr_instr(tape)?;
                    let curr_instr = curr_multi_instr.instr;

                    if curr_instr == Instr::StartLoop && depth == 0 { break; }

                    if curr_instr == Instr::EndLoop {
                        depth += 1;
                    } else if curr_instr == Instr::StartLoop {
                        depth -= 1;
                    }
                }
            },
            Instr::Increment => { self.incr_instr(); }
            Instr::Decrement => { self.incr_instr(); }
            Instr::Input => { self.incr_instr(); }
            Instr::Output => { self.incr_instr(); }
            Instr::Fork => { self.incr_instr(); }
        }

        // Save any output
        let output = match instr {
            Instr::Output => {
                let val = mget_def(&curr_memory.0, 0, self.memory_ptr.clone());
                let text = if ascii_mode {
                    // TODO: Handle chars in range > 2^31 expressible by i32
                    match std::char::from_u32(val as u32) {
                        None => format!("{}", val),
                        Some(c) => format!("{}", c)
                    }
                } else {
                    format!("{}", val)
                };
                Some(text)
            },
            _ => None
        };

        // Calculate the memory modifying function
        let memory_edit = match instr {
            Instr::Input => {
                let raw_inp = inputter.recv(&self);
                let inp: i32 = raw_inp.trim().parse().expect("Non-number input from input!");
                MemoryEdit::Absolute(self.memory_ptr.clone(), inp)
            },
            Instr::Increment =>
                MemoryEdit::Delta(self.memory_ptr.clone(), 1),
            Instr::Decrement =>
                MemoryEdit::Delta(self.memory_ptr.clone(), -1),
            _ =>
                MemoryEdit::NoEdit
        };

        let will_fork = instr == Instr::Fork;

        self.steps_taken += 1;
        Some((output, memory_edit, will_fork))
    }

    fn curr_cycle (&self) -> usize {
         self.started_at + self.steps_taken
    }
}

#[derive(Debug, Clone)]
enum MemoryEdit {
    Delta(Address, i32),
    Absolute(Address, i32),
    NoEdit
}

// map utils
fn mget_def<K, V> (m: &BTreeMap<K, V>, def: V, key: K) -> V
    where
        K: Ord,
        V: Copy + Eq
{
    *m.get(&key).unwrap_or(&def)
}

fn mset_def<K, V> (m: &mut BTreeMap<K, V>, def: V, remove_default: bool, key: K, new_val: V) -> ()
    where
        K: Ord,
        V: Copy + Eq
{
    if new_val == def && remove_default {
        m.remove(&key);
    } else {
        m.insert(key, new_val);
    }
}

fn mmod_map<K, V, F> (m: &mut BTreeMap<K, V>, def: V, remove_default: bool, key: K, f: F)
    where
        K: Ord + Clone,
        V: Copy + Eq,
        F: FnOnce(V) -> V
{
    mset_def(m, def, remove_default, key.clone(), f(mget_def(m, def, key)));
}

// cursive utils

struct PercentSizeView<V> {
    subview: V,
    percent: XY<(f32, f32)>
}

impl<V> PercentSizeView<V> {
    fn new (percent: XY<(f32, f32)>, subview: V) -> Self {
        PercentSizeView { percent, subview }
    }
}

impl<V: View> ViewWrapper for PercentSizeView<V> {
    type V = V;

    fn with_view<F: FnOnce(&Self::V) -> R, R> (&self, f: F) -> Option<R> {
        Some(f(&self.subview))
    }

    fn with_view_mut<F: FnOnce(&mut Self::V) -> R, R> (&mut self, f: F) -> Option<R> {
        Some(f(&mut self.subview))
    }

    fn wrap_required_size (&mut self, req: XY<usize>) -> XY<usize> {
        let x = (req.x as f32 * self.percent.x.1) as usize -
                (req.x as f32 * self.percent.x.0) as usize;

        let y = (req.y as f32 * self.percent.y.1) as usize -
                (req.y as f32 * self.percent.y.0) as usize;

        XY { x, y }
    }
}

trait PercentSizeable: View + Sized {
    fn percent_size (self: Self, percent: XY<(f32, f32)>) -> PercentSizeView<Self> {
        PercentSizeView::new(percent, self)
    }

    fn percent_height (self: Self, percent: (f32, f32)) -> PercentSizeView<Self> {
        self.percent_size(XY {
            x: (0.0, 1.0),
            y: percent
        })
    }

    fn percent_width (self: Self, percent: (f32, f32)) -> PercentSizeView<Self> {
        self.percent_size(XY {
            x: percent,
            y: (0.0, 1.0)
        })
    }
}

impl<T: View> PercentSizeable for T {}
