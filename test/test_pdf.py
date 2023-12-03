from scipy import integrate
import math
from src.distribution_utils.distributions import partial_pdf


def pdf_test_template(func, lim=(0, 1)):
    r"""Checks whether a pdf is normalized for a given function within some limits.

    @param func    A function, generating the pdf.
    @param lim     The bounds the probability function is defined in."""

    area = integrate.quad(func, *lim)[0]

    # Check if the area under the pdf curve is approximately 1.
    assert math.isclose(area, 1.0, rel_tol=1e-7)


def test_normalization():
    r"""! Tests whether pdf normalization is correct."""

    # Define p.d.f for some parameters
    pdf = partial_pdf(f=0, lam=1, mu=5.28, sigma=0.3, alpha=0, beta=10)
    pdf_test_template(pdf, lim=(0, 10))

    pdf = partial_pdf(f=0.1, lam=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6)
    pdf_test_template(pdf, lim=(5, 5.6))

    pdf = partial_pdf(f=0.5, lam=0.7, mu=3.2, sigma=0.6, alpha=4, beta=5.6)
    pdf_test_template(pdf, lim=(4, 5.6))

    pdf = partial_pdf(f=0.99, lam=10, mu=-10, sigma=2.3, alpha=4, beta=8)
    pdf_test_template(pdf, lim=(4, 8))
