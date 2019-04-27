initSidebarItems({"enum":[["ArgminError",""],["CheckpointMode",""],["ObserverMode","This is used to indicate how often the observer will observe the status. `Never` deactivates the observer, `Always` and `Every(i)` will call the observer in every or every ith iteration, respectively. `NewBest` will call the observer only, if a new best solution is found."],["TerminationReason","Indicates why the optimization algorithm stopped"]],"fn":[["load_checkpoint",""]],"macro":[["check_param","Release an `T` from an `Option<T>` if it is not `None`. If it is `None`, return an `ArgminError` with a message that needs to be provided."],["make_kv","Creates an `ArgminKV` at compile time in order to avoid pushing to the `kv` vector."],["trait_bound","Reuse a list of trait bounds by giving it a name, e.g. trait_bound!(CopyAndDefault; Copy, Default);"]],"mod":[["executor","Executor"],["file","Output parameter vectors to file"],["finitediff","Finite Differentiation"],["macros","Macros # Macros"],["modcholesky","Modified Cholesky decompositions Modified Cholesky decompositions"],["slog_logger","Loggers based on the `slog` crate"]],"struct":[["ArgminCheckpoint",""],["ArgminIterData","The datastructure which is returned by the `next_iter` method of the `Solver` trait."],["ArgminKV","A simple key-value storage"],["ArgminResult","This is returned by the `Executor` after the solver is run on the operator."],["Error","The `Error` type, which can contain any failure."],["IterState",""],["MinimalNoOperator",""],["NoOperator","Fake Operators for testing No-op operator with free choice of the types"],["Observer","Container for observers which acts just like a single `Observe`r by implementing `Observe` on it."],["OpWrapper","This wraps an operator and keeps track of how often the cost, gradient and Hessian have been computed and how often the modify function has been called. Usually, this is an implementation detail unless a solver is needed within another solver (such as a line search within a gradient descent method), then it may be necessary to wrap the operator in an OpWrapper."]],"trait":[["ArgminAdd","Add a `T` to `self`"],["ArgminDiv","(Pointwise) Divide a `T` by `self`"],["ArgminDot","Dot/scalar product of `T` and `self`"],["ArgminEye",""],["ArgminInv","Compute the inverse (`T`) of `self`"],["ArgminLineSearch","Defines a common interface for line search methods."],["ArgminMinMax",""],["ArgminMul","(Pointwise) Multiply a `T` with `self`"],["ArgminNLCGBetaUpdate","Common interface for beta update methods (Nonlinear-CG)"],["ArgminNorm","Compute the l2-norm (`U`) of `self`"],["ArgminOp","This trait needs to be implemented for every operator/cost function."],["ArgminRandom",""],["ArgminScaledAdd","Add a `T` scaled by an `U` to `self`"],["ArgminScaledSub","Subtract a `T` scaled by an `U` from `self`"],["ArgminSub","Subtract a `T` from `self`"],["ArgminTranspose",""],["ArgminTrustRegion","Defines a common interface to methods which calculate approximate steps for trust region methods."],["ArgminWeightedDot","Dot/scalar product of `T` and `self` weighted by W (p^TWv)"],["ArgminZero","Return param vector of all zeros (for now, this is a hack. It should be done better)"],["ArgminZeroLike","Zero for dynamically sized objects"],["Observe","Defines the interface every Observer needs to expose"],["Solver",""]]});