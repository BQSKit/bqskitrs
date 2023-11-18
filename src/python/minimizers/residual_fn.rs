use crate::{
    ir::circuit::Circuit,
    ir::inst::minimizers::{
        CostFn, DifferentiableResidualFn, HilbertSchmidtResidualFn, ResidualFn, ResidualFunction, HilbertSchmidtStateResidualFn, HilbertSchmidtSystemResidualFn,
    },
};
use ndarray::Array2;
use ndarray_linalg::c64;
use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyTuple};

struct PyResidualFn {
    cost_fn: PyObject,
}

impl PyResidualFn {
    pub fn new(cost_fn: PyObject) -> Self {
        PyResidualFn { cost_fn }
    }
}

impl CostFn for PyResidualFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_cost", args) {
            Ok(val) => val
                .extract::<f64>(py)
                .expect("Return type of get_cost was not a float."),
            Err(..) => panic!("Failed to call 'get_cost' on passed ResidualFunction."), // TODO: make a Python exception?
        }
    }
}

impl ResidualFn for PyResidualFn {
    fn num_residuals(&self) -> usize {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self.cost_fn.call_method0(py, "num_residuals") {
            Ok(val) => val
                .extract::<usize>(py)
                .expect("Return of num_residuals was not an integer."),
            Err(..) => panic!("Failed to call num_residuals on passed residual function."),
        }
    }

    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_residuals", args) {
            Ok(val) => val
                .extract::<Vec<f64>>(py)
                .expect("Return type of get_residuals was not a sequence of floats."),
            Err(e) => {
                println!("{:?}, {:?}, {:?}", e.get_type(py), e.value(py), e.traceback(py));
                panic!("Failed to call 'get_residuals' on passed ResidualFunction."); // TODO: make a Python exception?
            },
        }
    }
}

impl DifferentiableResidualFn for PyResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        let pyarray = match self.cost_fn.call_method1(py, "get_grad", args) {
            Ok(val) => val
                .extract::<Py<PyArray2<f64>>>(py)
                .expect("Return type of get_grad was not a matrix of floats."),
            Err(..) => panic!("Failed to call 'get_grad' on passed ResidualFunction."), // TODO: make a Python exception?
        };
        pyarray.into_ref(py).to_owned_array()
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        (self.get_residuals(params), self.get_grad(params))
    }
}

#[pyclass(
    name = "HilbertSchmidtResidualsFunction",
    subclass,
    unsendable,
    module = "bqskitrs"
)]
pub struct PyHilberSchmidtResidualFn {
    cost_fn: ResidualFunction,
}

#[pymethods]
impl PyHilberSchmidtResidualFn {
    #[new]
    pub fn new(circ: Circuit, target_matrix: &PyAny) -> PyResult<Self> {
        let cls = target_matrix.getattr("__class__")?;
        let dunder_name = cls.getattr("__name__")?;
        let name = dunder_name.extract::<&str>()?;
        let costfn = match name {
            "UnitaryMatrix" => {
                let np = target_matrix
                    .getattr("numpy")?
                    .extract::<&PyArray2<c64>>()?;
                ResidualFunction::HilbertSchmidt(Box::new(HilbertSchmidtResidualFn::new(circ, np.to_owned_array())))
            }
            "StateVector" => {
                let np = target_matrix
                    .getattr("numpy")?
                    .extract::<&PyArray1<c64>>()?;
                ResidualFunction::HilbertSchmidtState(Box::new(HilbertSchmidtStateResidualFn::new(circ, np.to_owned_array())))
            }
            "StateSystem" => {
                let np = target_matrix
                    .getattr("target")?
                    .extract::<&PyArray2<c64>>()?;
                let vec_count = target_matrix.getattr("_vec_count")?.extract::<u32>()?;
                ResidualFunction::HilbertSchmidtSystem(Box::new(HilbertSchmidtSystemResidualFn::new(circ, np.to_owned_array(), vec_count)))
            }
            "ndarray" => {
                let np = target_matrix
                    .extract::<&PyArray2<c64>>()?;
                ResidualFunction::HilbertSchmidt(Box::new(HilbertSchmidtResidualFn::new(circ, np.to_owned_array())))
            }
            _ => panic!("HilbertSchmidtCost only takes numpy arrays or UnitaryMatrix types."),
        };
        Ok(PyHilberSchmidtResidualFn {cost_fn: costfn})
    }

    pub fn __call__(&self, py: Python, params: Vec<f64>) -> Vec<f64> {
        self.get_residuals(py, params)
    }

    pub fn get_cost(&self, _py: Python, params: Vec<f64>) -> f64 {
        self.cost_fn.get_cost(&params)
    }

    pub fn get_residuals(&self, _py: Python, params: Vec<f64>) -> Vec<f64> {
        self.cost_fn.get_residuals(&params)
    }

    pub fn get_grad(&self, py: Python, params: Vec<f64>) -> Py<PyArray2<f64>> {
        PyArray2::from_array(py, &self.cost_fn.get_grad(&params)).to_owned()
    }

    pub fn get_residuals_and_grad(
        &self,
        py: Python,
        params: Vec<f64>,
    ) -> (Vec<f64>, Py<PyArray2<f64>>) {
        let (residuals, grad) = self.cost_fn.get_residuals_and_grad(&params);
        (residuals, grad.into_pyarray(py).to_owned())
    }
}

fn is_cost_fn_obj(obj: &'_ PyAny) -> PyResult<bool> {
    if obj.hasattr("get_cost")? {
        let get_cost = obj.getattr("get_cost")?;
        let get_residuals = obj.getattr("get_residuals")?;
        if get_cost.is_callable()
            && get_residuals.is_callable()
            && obj.hasattr("get_grad")?
            && obj.getattr("get_grad")?.is_callable()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

impl<'source> FromPyObject<'source> for ResidualFunction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match ob.extract::<Py<PyHilberSchmidtResidualFn>>() {
            Ok(fun) => {
                let costfn = &fun.try_borrow(py)?.cost_fn;
                match costfn {
                    ResidualFunction::HilbertSchmidt(hs) => Ok(ResidualFunction::HilbertSchmidt(hs.clone())),
                    ResidualFunction::HilbertSchmidtState(hs) => Ok(ResidualFunction::HilbertSchmidtState(hs.clone())),
                    ResidualFunction::HilbertSchmidtSystem(hs) => Ok(ResidualFunction::HilbertSchmidtSystem(hs.clone())),
                    _ => panic!("Unexpected dynamic cost function."),
                }
            },
            Err(..) => {
                if is_cost_fn_obj(ob)? {
                    let fun = PyResidualFn::new(ob.into());
                    Ok(ResidualFunction::Dynamic(Box::new(fun)))
                } else {
                    panic!("Failed to extract ResidualFn from obj."); // TODO: throw a Python error here.
                }
            }
        }
    }
}
