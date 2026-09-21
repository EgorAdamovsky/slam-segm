//! Данный модуль предоставляет абстракции для построения многопоточной цепочки обработки данных
//!
//! Цепочка представлена типом Pipeline, объединяющая несколько узлов Node, каждый из которых выполняет в отдельном потоке модуль Module.
//!
//! Каждый модуль имеет ассоциированные типы In и Out, а также метод process. Если тип Out одного модуля совпадает с типом In другого, их можно связать:
//!
//! ```
//! use node_plumbing::{Pipeline, Module, Output};
//!
//! struct SourceModule {
//!     num: u32,
//! }
//!
//! impl Module for SourceModule {
//!     type In = ();
//!
//!     type Out = u32;
//!
//!     fn process(&mut self, _input: Self::In) -> eyre::Result<Self::Out> {
//!         self.num += 1;
//!         Ok(self.num)
//!     }
//! }
//!
//! struct MultiplierModule;
//!
//! impl Module for MultiplierModule {
//!     type In = u32;
//!     type Out = u64;
//!     fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
//!         Ok(input as u64 * 2)
//!     }
//! }
//! let pipeline = Pipeline::new();
//! let mut out = {
//!     let mut source = pipeline.source_node(SourceModule { num: 0 });
//!     let mut multiplier = pipeline.node(MultiplierModule, source.output());
//!     multiplier.output().linked_input()
//! };
//! pipeline.start();
//! ```

use std::{
    any::{Any, type_name},
    cell::RefCell,
    marker::{PhantomData, Send},
    ops::{Deref, DerefMut},
    sync::mpsc,
    thread,
};

use eyre::{Context, bail};
use tracing::{info, warn};

// struct NodeInfo {}

pub struct AddNodeOnDrop<'a, M: Module + 'static, I: Input>(&'a Pipeline, Option<Node<M, I>>)
where
    I: Input<Item = M::In> + Send + 'static,
    <M as Module>::In: Sync,
    <M as Module>::In: Send,
    <M as Module>::Out: Sync,
    <M as Module>::Out: Send;

impl<M: Module + 'static, I: Input> Drop for AddNodeOnDrop<'_, M, I>
where
    I: Input<Item = M::In> + Send + 'static,
    <M as Module>::In: Sync,
    <M as Module>::In: Send,
    <M as Module>::Out: Sync,
    <M as Module>::Out: Send,
{
    fn drop(&mut self) {
        self.0
            .nodes
            .borrow_mut()
            .push(Box::new(self.1.take().unwrap()));
    }
}

impl<M: Module + 'static, I: Input> Deref for AddNodeOnDrop<'_, M, I>
where
    I: Input<Item = M::In> + Send + 'static,
    <M as Module>::In: Sync,
    <M as Module>::In: Send,
    <M as Module>::Out: Sync,
    <M as Module>::Out: Send,
{
    type Target = Node<M, I>;

    fn deref(&self) -> &Self::Target {
        self.1.as_ref().unwrap()
    }
}

impl<M: Module + 'static, I: Input> DerefMut for AddNodeOnDrop<'_, M, I>
where
    I: Input<Item = M::In> + Send + 'static,
    <M as Module>::In: Sync,
    <M as Module>::In: Send,
    <M as Module>::Out: Sync,
    <M as Module>::Out: Send,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.1.as_mut().unwrap()
    }
}

pub struct Pipeline {
    nodes: RefCell<Vec<Box<dyn DynNode>>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline {
            nodes: RefCell::new(Vec::new()),
        }
    }

    pub fn source_node<'out, 'pipe, M: Module<In = ()> + 'static>(
        &'pipe self,
        module: M,
    ) -> AddNodeOnDrop<'pipe, M, InputDefault<()>>
    where
        <M as Module>::In: Sync,
        <M as Module>::In: Send,
        <M as Module>::Out: Sync,
        <M as Module>::Out: Send,
    {
        AddNodeOnDrop(self, Some(Node::new_source(module)))
    }

    pub fn node<'out, 'pipe, M: Module + 'static, I: Input + Send, O>(
        &'pipe self,
        module: M,
        prev_output: &'out mut O,
    ) -> AddNodeOnDrop<'pipe, M, I>
    where
        O: Output<Item = M::In, LinkedInput = I> + 'out,
        I: Input<Item = M::In> + Send + 'static,
        <M as Module>::In: Sync,
        <M as Module>::In: Send,
        <M as Module>::Out: Sync,
        <M as Module>::Out: Send,
    {
        AddNodeOnDrop(self, Some(Node::new(module, prev_output)))
    }

    pub fn start(self) {
        let Self { nodes } = self;
        let nodes = nodes.into_inner();
        for node in nodes {
            node.start();
        }
    }
}

#[must_use]
pub struct Node<M: Module, I: Input> {
    module: M,
    input: I,
    output: OutputChannel<M::Out>,
}

pub struct OutputChannel<T> {
    queues: Vec<mpsc::SyncSender<T>>,
    level: usize,
}

#[derive(Default)]
pub struct InputDefault<T>(PhantomData<T>);

pub struct InputChannel<T> {
    source_queue: mpsc::Receiver<T>,
}

pub struct InputCombined<T> {
    source_inputs: T,
}

pub trait Module: Send {
    type In;
    type Out: Clone;

    fn name(&self) -> &'static str {
        type_name::<Self>()
    }
    fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out>;
}

pub trait Input {
    type Item;
    fn recv(&mut self) -> eyre::Result<Self::Item>;
}

pub trait Output {
    type Item;
    type LinkedInput: Input<Item = Self::Item>;

    fn level(&self) -> usize;
    fn linked_input_for_level(&mut self, for_level: usize) -> Self::LinkedInput;
    fn linked_input(&mut self) -> Self::LinkedInput {
        self.linked_input_for_level(self.level() + 1)
    }
}

impl<M> Node<M, InputDefault<()>>
where
    M: Module<In = ()> + 'static,
{
    pub(crate) fn new_source(module: M) -> Self {
        Self::new_from_input(module, InputDefault::default(), 1)
    }
}

impl<M, I> Node<M, I>
where
    M: Module + 'static,
    I: Input<Item = M::In> + Send + 'static,
{
    fn new_from_input(module: M, input: I, level: usize) -> Self {
        Self {
            module,
            output: OutputChannel::new(level),
            input,
        }
    }

    fn new<'a, O>(module: M, prev_output: &'a mut O) -> Self
    where
        O: Output<Item = M::In, LinkedInput = I> + 'a,
    {
        Self::new_from_input(
            module,
            prev_output.linked_input_for_level(prev_output.level() + 1),
            prev_output.level() + 1,
        )
    }

    pub fn output(&mut self) -> &mut OutputChannel<M::Out> {
        &mut self.output
    }

    fn run(mut self) -> eyre::Result<()> {
        loop {
            let input = self.input.recv()?;
            let output = self
                .module
                .process(input)
                .wrap_err("Failed to execute Module::process")?;
            if !self.output.send(output) {
                break;
            }
        }
        Ok(())
    }
}

trait DynNode: Any {
    fn start(self: Box<Self>);
}

impl<M, I> DynNode for Node<M, I>
where
    M: Module + 'static,
    I: Input<Item = M::In> + Send + 'static,
    <M as Module>::In: Sync,
    <M as Module>::In: Send,
    <M as Module>::Out: Sync,
    <M as Module>::Out: Send,
{
    fn start(self: Box<Self>) {
        thread::spawn(move || {
            // TODO better error reporting.
            let name = self.module.name();
            if let Err(err) = self.run() {
                warn!("Node {name} has stopped: {:?}", err);
            } else {
                info!("Node {name} has stopped, as all outputs have been closed");
            }
        });
    }
}

impl<T: Default> Input for InputDefault<T> {
    type Item = T;

    fn recv(&mut self) -> eyre::Result<T> {
        Ok(T::default())
    }
}

impl<T> Input for InputChannel<T> {
    type Item = T;

    fn recv(&mut self) -> eyre::Result<T> {
        Ok(self
            .source_queue
            .recv()
            .wrap_err("Failed to receive input (previous Node shut down?)")?)
    }
}

impl<T> InputChannel<T> {
    pub fn try_recv(&mut self) -> eyre::Result<Option<T>> {
        match self.source_queue.try_recv() {
            Ok(val) => Ok(Some(val)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                bail!("Failed to receive input (previous Node shut down?)")
            }
        }
    }
}

impl<T> InputCombined<T> {
    pub(crate) fn new(inputs: T) -> Self {
        Self {
            source_inputs: inputs,
        }
    }
}

#[crabtime::function]
fn gen_combined_io(n: usize) {
    for count in 2..=n {
        let generics = (0..count).map(|i| format!("T{i}")).collect::<Vec<_>>();
        let generics_cs = generics.join(", ");
        let generics_cs_mut_ref = generics
            .iter()
            .map(|g| format!("&mut {g}"))
            .collect::<Vec<_>>()
            .join(", ");
        let generics_inputs = generics
            .iter()
            .map(|g| format!("{g}: Input"))
            .collect::<Vec<_>>()
            .join(", ");
        let generics_outputs = generics
            .iter()
            .map(|g| format!("{g}: Output"))
            .collect::<Vec<_>>()
            .join(", ");
        let generics_items = generics
            .iter()
            .map(|g| format!("{g}::Item"))
            .collect::<Vec<_>>()
            .join(", ");
        let generics_linked_inputs = generics
            .iter()
            .map(|g| format!("{g}::LinkedInput"))
            .collect::<Vec<_>>()
            .join(", ");
        let generics_ret = generics
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let msg = format!("\"Failed to get value from input {g}\"");
                crabtime::quote! {
                    self
                        .source_inputs
                        .{{i}}
                        .recv()
                        .wrap_err({{msg}})?
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let linked_inputs = generics
            .iter()
            .enumerate()
            .map(|(i, _g)| {
                crabtime::quote! {
                    self.{{i}}.linked_input_for_level(for_level)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let levels_of_inputs = generics
            .iter()
            .enumerate()
            .map(|(i, _g)| {
                crabtime::quote! {
                    self.{{i}}.level()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        crabtime::output! {
            impl<{{generics_inputs}}> Input for InputCombined<({{generics_cs}})> {
                type Item = ({{generics_items}});

                fn recv(&mut self) -> eyre::Result<Self::Item> {
                    Ok(({{generics_ret}}))
                }
            }

            impl<{{generics_outputs}}> Output for ({{generics_cs_mut_ref}}) {
                type Item = ({{generics_items}});
                type LinkedInput = InputCombined<({{generics_linked_inputs}})>;

                fn level(&self) -> usize {
                    *[{{levels_of_inputs}}].iter().max().unwrap()
                }

                fn linked_input_for_level(&mut self, for_level: usize) -> Self::LinkedInput {
                    InputCombined::new((
                        {{linked_inputs}}
                    ))
                }
            }
        };
    }
}

// Генерирует реализации Input и Output для кортежей из нескольких Output
gen_combined_io!(8);

impl<T: Clone> OutputChannel<T> {
    fn new(level: usize) -> Self {
        Self {
            queues: Default::default(),
            level,
        }
    }

    fn send(&self, data: T) -> bool {
        let mut any_ok = false;
        for queue in &self.queues {
            any_ok = queue.send(data.clone()).is_ok() || any_ok;
        }
        any_ok
    }
}

impl<T> Output for OutputChannel<T> {
    type Item = T;
    type LinkedInput = InputChannel<T>;

    fn level(&self) -> usize {
        self.level
    }

    fn linked_input_for_level(&mut self, for_level: usize) -> Self::LinkedInput {
        let (s, r) = mpsc::sync_channel(for_level - self.level - 1);
        self.queues.push(s);
        InputChannel { source_queue: r }
    }
}

#[cfg(test)]
mod test {
    use crate::{Input, Module, Output, Pipeline};

    struct SourceModule {
        num: u32,
    }

    impl Module for SourceModule {
        type In = ();

        type Out = u32;

        fn process(&mut self, _input: Self::In) -> eyre::Result<Self::Out> {
            self.num += 1;
            Ok(self.num)
        }
    }

    struct MultiplierModule;

    impl Module for MultiplierModule {
        type In = u32;

        type Out = u64;

        fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
            Ok(input as u64 * 2)
        }
    }

    #[test]
    fn nodestream_basic() {
        let pipeline = Pipeline::new();

        let mut out = {
            let mut source = pipeline.source_node(SourceModule { num: 0 });
            let mut multiplier = pipeline.node(MultiplierModule, source.output());
            assert_eq!(source.output.level, 1);
            assert_eq!(multiplier.output.level, 2);
            multiplier.output().linked_input()
        };

        pipeline.start();
        assert_eq!(out.recv().unwrap(), 2);
        assert_eq!(out.recv().unwrap(), 4);
    }
}
