use std::ops::Index;

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f64>,
}

impl Chromosome {
    pub fn new(genes: Vec<f64>) -> Self {
        Self { genes }
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl IntoIterator for Chromosome {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

impl FromIterator<f64> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}
