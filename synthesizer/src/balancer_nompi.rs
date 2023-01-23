use std::thread::{spawn, JoinHandle};

type Handles<T> = Vec<JoinHandle<T>>;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer<T> {
    pub workers: usize,
    pub rank: usize,
    handles: Handles<T>,
}

#[cfg(not(feature = "balancer"))]
impl<T> Balancer<T> {
    /// Constructs a new `Balancer` from an `mpi::SystemCommunicator` a.k.a. `world`.
    pub fn new(workers: usize) -> Self {
        Balancer {
            workers,
            rank: 0,
            handles: vec![],
        }
    }

    /// Calculates local set of items on which to work on.
    pub fn local_set<I: Copy + Clone>(&self, items: &Vec<I>) -> Vec<I> {
        // Gather and return local set of items
        items.clone()
    }

    /// Adds a handle
    pub fn add(&mut self, handle: JoinHandle<T>) {
        self.wait_limit();
        self.handles.push(handle);
    }

    /// Adds a handle
    pub fn spawn<F>(&mut self, f: F)
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        self.wait_limit();
        self.handles.push(spawn(f));
    }

    /// Waits for all threads to finish (only on this rank! see `barrier` for blocking across all ranks).
    pub fn wait(&mut self) {
        while self.handles.len() > 0 {
            semi_spinlock();
            self.handles.retain(|task| !task.is_finished());
        }
    }

    /// Waits for all threads to finish (across all ranks! see `barrier` for blocking on one rank).
    pub fn barrier(&mut self) {
        self.wait();
    }

    /// Wait until there is a free worker on this rank
    fn wait_limit(&mut self) {
        while self.handles.len() >= self.workers {
            semi_spinlock();
            self.handles.retain(|task| !task.is_finished());
        }
    }
}

const SEMI_SPINLOCK: u64 = 10;
fn semi_spinlock() {
    std::thread::sleep(std::time::Duration::from_millis(SEMI_SPINLOCK))
}
