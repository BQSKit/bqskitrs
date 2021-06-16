#pragma once

#include <ceres/c_api.h>
#include <ceres/ceres.h>


inline void ceres_solve_silent(ceres_problem_t *c_problem, size_t max_iters, size_t num_threads, double ftol, double gtol) {
    ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);

    ceres::Solver::Options options;
    options.max_num_iterations = max_iters;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = num_threads;
    options.minimizer_progress_to_stdout = false;
    options.function_tolerance = ftol;
    options.gradient_tolerance = gtol;
    // Ceres outputs a *lot* of logs, so we silence them here for our own uses
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    // good for debugging
    //std::cout << summary.FullReport() << "\n";
}
