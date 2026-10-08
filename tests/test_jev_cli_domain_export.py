"""The jev-CLI domain is loadable by module name, like any domain module."""

from jev_factory.domain.model import Domain
from jev_factory.domain.validate import load_domain
from jev_factory.domains.jev_cli.generate import generate_domain


def test_the_package_exposes_the_generated_domain_by_name():
    domain = load_domain("jev_factory.domains.jev_cli")
    assert isinstance(domain, Domain)
    assert domain.names() == generate_domain().names()


def test_the_domain_is_built_lazily_on_first_access():
    import jev_factory.domains.jev_cli as pkg

    assert "DOMAIN" not in vars(pkg) or isinstance(vars(pkg)["DOMAIN"], Domain)
    assert isinstance(pkg.DOMAIN, Domain)
