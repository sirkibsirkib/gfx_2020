#[derive(Debug)]
pub struct SimpleArena<T> {
    // invariant: for i in hole_index_stack { assert!(slots.get(i).unwrap().is_none()) }
    slots: Vec<Option<T>>,
    hole_index_stack: Vec<usize>,
}

impl<T> Default for SimpleArena<T> {
    fn default() -> Self {
        Self { slots: Default::default(), hole_index_stack: Default::default() }
    }
}

impl<T> SimpleArena<T> {
    pub fn add(&mut self, t: T) -> usize {
        if let Some(i) = self.hole_index_stack.pop() {
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            *slot = Some(t);
            i
        } else {
            self.slots.push(Some(t));
            self.slots.len() - 1
        }
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        let t = self.slots.get_mut(index)?.take()?;
        if index + 1 == self.slots.len() {
            self.slots.pop();
        } else {
            self.hole_index_stack.push(index);
        }
        Some(t)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index)?.as_ref()
    }
}

impl<T> IntoIterator for SimpleArena<T> {
    type Item = T;
    type IntoIter = std::iter::Flatten<std::vec::IntoIter<Option<T>>>;
    fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        self.slots.into_iter().flatten()
    }
}
