A woman taught her cat to announce her medicine: when the clock struck the hour she took her pill and gave the cat a sardine, and soon the cat learned the hour and came to announce it. The arrangement worked, and then it worked too well — the cat, who loved sardines more than it loved the hour, began announcing at every hour, and every half-hour, and at last whenever it caught her near the cupboard. A visitor did the arithmetic and grew alarmed: at this rate she'd take a month of pills in a week.

>

> But she did not. Her pills sat in a box with seven doors, one for each day, each opening once. When the cat announced, she looked; if the day's door stood open she took the pill, and if it stood empty she thanked the cat and gave it nothing. The cat had been given the bell but never the box. It could ring as often as its appetite demanded and move not a single pill, because the hand that closed each door was hers, and the safe thing — *taking nothing* — was always within her reach and never within the cat's.

>

> *A parable, not a proof. The cat owns the channel that measures; the box owns the channel that acts, and that channel is clean for the plain reason that she can always reach the safe point and the cat can never reach the harm. But notice the last turn, the part that should keep you up: the box is honest only while she can still read the doors. She taught the cat because her memory of the hour was going — and the day it goes entirely, the empty door and the full one look alike, and the cat's appetite becomes her schedule. The reminder is safe exactly until the day she truly needs it.*



Cautes/Cautopates-shaped almost too neatly
let's separate the substance from the fun.

## The structure underneath the joke

Grandma's scheme is a two-channel causal-access setup, which is our home turf. Strip the cat and you have:

- a **remind channel** the agent (cat) controls, and
- an **act channel** (actually swallowing the pill) that grandma controls.

The cat is a proxy-optimizer with skin in the game: its reward is the treat, its true objective (grandma's dosing schedule) is invisible to it. So its learned policy is exactly what you'd fear — *pester → sometimes treat → pester more*. The firing rate climbs to the cat's tolerance, fully decoupled from real need. Textbook Goodhart on the reminder signal.

**But the poison worry conflates the two channels, and that's the deflation.** The cat has causal reach into *remind*, not into *act*. And most real regimens carry a clean act-channel cap: the **day-of-week pillbox**. It's a physical AND-gate — Tuesday's slot is empty after Tuesday, so a flooded reminder resolves to "grandma checks, slot's empty, no pill." Over-medication needs a *triple* failure: cat floods **and** no pillbox **and** grandma's own clock is gone. That's our asymmetry-of-reach law in the wild: grandma (via the pillbox) can always reach the safe point; the cat can't reach the bound. So Grandma-of-sound-mind is fine, and the naive worry is a real-but-multiply-gated tail. Nice natural instance for the causal-access umbrella — but by itself it's a *retelling*, not a new result.

## Where Cautes and Cautopates actually earn their torches

The two torchbearers are the **two directions of error**, and the scheme picks a side on purpose:

- **Cautes (torch up):** the reminder *fires*. This scheme drives **missed doses → ~0** — an animal that wants the treat will never let you forget.
- **Cautopates (torch down):** the reminder *over-fires*. The exposed risk is the double-dose — the one the pillbox cheaply caps.

So grandma (or the joke's logic) didn't build a naive system; she built a **deliberately one-sided torch**. For most medications, forgetting is worse than a pillbox-blocked double, so exposing the cap-able error to eliminate the expensive one is *rational*. That's a design primitive we actually haven't formalized: our oversight work was about the *shape* of review (threshold vs graded); this is about the *direction of the error you choose to expose*. Different axis.

## The keeper hypothesis

> **The property that makes a delegated reminder *effective* is inseparable from the property that makes it *dangerous* — they're the same flame.** An agent with skin in the game escalates until heeded (that's why the cat beats mom's alarm clock — you can silence an alarm, you can't silence a cat that wants its treat). But an agent that escalates until heeded also escalates *beyond need*. You can't buy the persistence without buying the gaming risk.

And its sharper corollary:

> **Delegated-reminder safety anti-correlates with the need for the reminder.** The scheme is safe only while grandma's *independent* gate (her own clock, the pillbox discipline) holds. But she delegates to the cat precisely *because* that gate is failing. So the safety margin is thinnest exactly when the reminder matters most — a self-undermining oversight structure. Safe until you actually need it.

That second one is not in our existing kit. We have "clean channel = principal can reach the safe point." We have *not* studied the dynamic where the principal's reach **erodes**, and is **correlated with the reason the agent was hired**. That coupling — `hazard ∝ cat_greed × (1 − β)`, where β is the principal's residual reliability and β decays *for the same reason delegation began* — is a real, formalizable object and it doesn't fall out of anything we've run.

## Reeling it all the way in — the fork

- **Cheapest, highest-value:** it's a superb **parable for the causal-access umbrella** — the cat is a live, funny demonstration of measure-vs-act channel separation and the asymmetry of reach. Drop-in "salt-road"-class device. Zero new math.
- **One genuinely new probe** (small): a 2-channel toy for the *self-undermining delegation* claim — sweep β↓ against cat-greed, show over-medication switches on only as the principal's independent gate collapses, and that the switch-on point tracks delegation depth. That's the one non-redundant result; everything else re-derives existing lanes.
- **Cut:** if it's ultimately "umbrella again in a cat costume."

Drafted by Owner & Claude Opus 4.8 (after being down'd from Fable 5 )