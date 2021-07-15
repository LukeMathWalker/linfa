use super::{DatasetBase, Float, Records};
use ndarray::{ArrayBase, Axis, Data, Dimension};

#[cfg(feature = "polars")]
use polars_core::frame::DataFrame;

/// Implement records for NdArrays
impl<F: Float, S: Data<Elem = F>, I: Dimension> Records for ArrayBase<S, I> {
    type Elem = F;

    fn nsamples(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn nfeatures(&self) -> usize {
        self.len_of(Axis(1))
    }
}

/// Implement records for a DatasetBase
impl<F: Float, D: Records<Elem = F>, T> Records for DatasetBase<D, T> {
    type Elem = F;

    fn nsamples(&self) -> usize {
        self.records.nsamples()
    }

    fn nfeatures(&self) -> usize {
        self.records.nfeatures()
    }
}

/// Implement records for an empty dataset
impl Records for () {
    type Elem = ();

    fn nsamples(&self) -> usize {
        0
    }

    fn nfeatures(&self) -> usize {
        0
    }
}

/// Implement records for references
impl<R: Records> Records for &R {
    type Elem = R::Elem;

    fn nsamples(&self) -> usize {
        (*self).nsamples()
    }

    fn nfeatures(&self) -> usize {
        (*self).nfeatures()
    }
}

#[cfg(feature = "polars")]
impl Records for DataFrame {
    type Elem = ();

    fn nsamples(&self) -> usize {
        self.shape().0
    }

    fn nfeatures(&self) -> usize {
        self.shape().1
    }
}
