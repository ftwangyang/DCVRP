# Word environment mapping and interpretation record

This file separates the attached document's environment specification from
the user's experimental acceptance request.  The source DOCX is read-only and
was not edited.

## Rules taken from the attached Word document

| Word requirement | Implemented rule |
|---|---|
| Total horizon 480, divided into 10 equal intervals | Boundaries are 0, 48, ..., 480 minutes |
| Requests are disclosed only at interval transitions | A request is visible when its rounded disclosure boundary is no later than the current boundary |
| Each interval is a static multi-vehicle VRP | The solver receives one immutable complete static planning state per boundary |
| Preserve completed/in-progress customers and destroy unstarted suffix | A customer leg is committed if its dispatch starts strictly before the next boundary; the rest is replanned |
| One trip per vehicle inside an interval | Each active vehicle has at most one customer prefix and one physical depot return in an interval |
| Capacity cannot be reset merely at a boundary | Capacity is restored only after the corresponding physical depot return |
| Cost is actual executed distance | Charged edges are committed customer edges plus executed depot returns; destroyed planned suffixes are not charged |
| Final interval produces the final solution | A final complete static solve is performed at T=480, without opening another disclosure interval |

## Rule taken from the user's request, not from the Word document

The user repeatedly required a genuine multi-vehicle comparison rather than a
single active vehicle hidden inside multiple empty route containers.  The Word
mode therefore requires a nonempty first-stage route for every vehicle when at
least m initial requests are visible.  Later stages use the vehicles' actual
availability and do not force busy vehicles to accept work.

## Ambiguities resolved

- The document contains both fixed-boundary language and a sentence about
  advancing when tasks finish.  Fixed 48-minute boundaries take precedence
  because the preserve/destroy rules and the worked example are defined at Ti.
- The document uses both `< Ti` and `<= Ti` for disclosure.  Boundary equality
  is included because the detailed disclosure sections repeatedly say
  `disclosure time <= Ti`.
- "No capacity reset" is interpreted as no clock-based reset.  A physical
  depot return is the only replenishment event.

## Result-integrity rule

No cost target or closeness-to-Greedy constraint is used inside any solver or
environment.  Paper Greedy rows are retained only as legacy external
references; scientific comparisons use the same-instance Word-environment
Greedy rerun.
