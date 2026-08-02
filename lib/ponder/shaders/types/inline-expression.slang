implementing types;

// immediately invoked block, see include/inline-expression.slang
public struct ImplInlineExpression {
	public __init() {}
};

public R operator+<R, F : IFunc<R>>(ImplInlineExpression, F f) {
	return f();
}
