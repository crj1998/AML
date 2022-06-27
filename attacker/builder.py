
from functools import partial

from attacker import pgd

attacks = {
    "PGD": pgd.rpgd,
    "TRADES": pgd.trades
}
def build(atk, **kwargs):
    assert atk in attacks, f"Unknown attack: {atk}."
    attacker = partial(attacks[atk], **kwargs)
    return attacker