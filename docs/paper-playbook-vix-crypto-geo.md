# Paper Playbook — VIX, Oil Shock, Crypto ETF Options

Generated UTC: 2026-03-08

## Objective
Paper-first deployment for high-volatility, geopolitics-driven sessions with strict risk governance.

## Regime
- Trigger context: elevated vol complex (UVXY/VXX/VIXY), oil shock confirmation (USO/BNO), weak broad beta.
- Tactical assumption: convexity and hedge structures outperform broad long beta during panic windows.

## Strategy lanes
1. **VIX options lane**
   - Focus expiries near event window.
   - Prioritize contracts with strong volume/OI and acceptable spreads.
   - Avoid very wide-spread tails even with high raw volume.

2. **VIX fund momentum lane**
   - Instruments: UVXY, VXX, VIXY, SVIX.
   - Entry gated after first 30 minutes of US cash open.
   - Risk cap per idea and lane cap enforced.

3. **Crypto ETF options lane**
   - Instruments: IBIT, ETHA, FETH, BITO.
   - Directional entries require momentum alignment.
   - Otherwise classify as long-vol/hedge and reduce size.

4. **Cross-asset geo hedge lane**
   - Oil/defense/safe-haven overlays (USO/BNO/XLE/LMT/RTX/GLD).

## Risk policy
- Paper-only.
- Daily stop: 2.5% NAV.
- Weekly stop: 6.0% NAV.
- Max concurrent risk: 2.0% NAV.
- No auto live promotion without statistical gate pass.
