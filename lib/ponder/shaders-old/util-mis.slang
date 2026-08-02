module "util-mis";
__exported import "util-types";


// balance heuristic weight for combining separate sampling techniques with
// different probability densities. Veach's MIS estimator.
public func misBalanceWeight(
	pdfA: f32,
	pdfB: f32,
) -> f32 {
	const f32 sum = pdfA + pdfB;
	return sum > 0.0f ? pdfA / sum : 0.0f;
}

