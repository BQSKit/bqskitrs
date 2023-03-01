use crate::{
    ir::circuit::Circuit,
    ir::inst::minimizers::{CostFn, CostFunction, DifferentiableCostFn, HilbertSchmidtCostFn, HilbertSchmidtStateCostFn},
};
use ndarray_linalg::c64;
use numpy::{PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyTuple};

struct PyCostFn {
    cost_fn: PyObject,
}

impl PyCostFn {
    pub fn new(cost_fn: PyObject) -> Self {
        PyCostFn { cost_fn }
    }
}

impl CostFn for PyCostFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_cost", args) {
            Ok(val) => val
                .extract::<f64>(py)
                .expect("Return type of get_cost was not a float."),
            Err(..) => panic!("Failed to call 'get_cost' on passed CostFunction."), // TODO: make a Python exception?
        }
    }
}

impl DifferentiableCostFn for PyCostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_grad", args) {
            Ok(val) => val
                .extract::<Vec<f64>>(py)
                .expect("Return type of get_grad was not a list of floats."),
            Err(..) => panic!("Failed to call 'get_grad' on passed CostFunction."), // TODO: make a Python exception?
        }
    }
}

#[pyclass(
    name = "HilbertSchmidtCostFunction",
    subclass,
    unsendable,
    module = "bqskitrs"
)]
pub struct PyHilberSchmidtCostFn {
    cost_fn: CostFunction,
}

#[pymethods]
impl PyHilberSchmidtCostFn {
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
                CostFunction::HilbertSchmidt(HilbertSchmidtCostFn::new(circ, np.to_owned_array()))
            }
            "StateVector" => {
                let np = target_matrix
                    .getattr("numpy")?
                    .extract::<&PyArray1<c64>>()?;
                CostFunction::HilbertSchmidtState(HilbertSchmidtStateCostFn::new(circ, np.to_owned_array()))
            }
            "StateSystem" => {
                let np = target_matrix
                    .getattr("target")?
                    .extract::<&PyArray2<c64>>()?;
                CostFunction::HilbertSchmidt(HilbertSchmidtCostFn::new(circ, np.to_owned_array()))
            }
            "ndarray" => {
                let np = target_matrix
                    .extract::<&PyArray2<c64>>()?;
                CostFunction::HilbertSchmidt(HilbertSchmidtCostFn::new(circ, np.to_owned_array()))
            }
            _ => panic!("HilbertSchmidtCost only takes numpy arrays or UnitaryMatrix types."),
        };
        Ok(PyHilberSchmidtCostFn {cost_fn: costfn})
    }

    pub fn __call__(&self, py: Python, params: Vec<f64>) -> f64 {
        self.get_cost(py, params)
    }

    pub fn get_cost(&self, _py: Python, params: Vec<f64>) -> f64 {
        self.cost_fn.get_cost(&params)
    }

    pub fn get_grad(&self, _py: Python, params: Vec<f64>) -> Vec<f64> {
        self.cost_fn.get_grad(&params)
    }

    pub fn get_cost_and_grad(&self, _py: Python, params: Vec<f64>) -> (f64, Vec<f64>) {
        self.cost_fn.get_cost_and_grad(&params)
    }
}

fn is_cost_fn_obj(obj: &'_ PyAny) -> PyResult<bool> {
    if obj.hasattr("get_cost")? {
        let get_cost = obj.getattr("get_cost")?;
        if get_cost.is_callable()
            && obj.hasattr("get_grad")?
            && obj.getattr("get_grad")?.is_callable()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

impl<'source> FromPyObject<'source> for CostFunction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match ob.extract::<Py<PyHilberSchmidtCostFn>>() {
            Ok(fun) => {
                let costfn = &fun.try_borrow(py)?.cost_fn;
                match costfn {
                    CostFunction::HilbertSchmidt(hs) => Ok(CostFunction::HilbertSchmidt(hs.clone())),
                    CostFunction::HilbertSchmidtState(hs) => Ok(CostFunction::HilbertSchmidtState(hs.clone())),
                    _ => panic!("Unexpected dynamic cost function."),
                }
            },
            Err(..) => {
                if is_cost_fn_obj(ob)? {
                    let fun = PyCostFn::new(ob.into());
                    Ok(CostFunction::Dynamic(Box::new(fun)))
                } else {
                    panic!("Failed to extract CostFn from obj."); // TODO: throw a Python error here.
                }
            }
        }
    }
}
