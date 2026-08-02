#pragma once

// let x = InlineExpr { let a = 4; return a; };
// captures are read-only copies
#define InlineExpr ImplInlineExpression() + () =>
