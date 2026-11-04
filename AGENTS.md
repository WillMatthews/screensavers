# Repository guidance

- Always keep `README.md` up to date with changes to screensavers, the active
  and benched lists, launch/build instructions, and multi-monitor behaviour.
- Every new screensaver must have a current screenshot in `docs/img/`, embedded
  in its README entry. Refresh screenshots when the appearance changes.
- Include the README and screenshot updates in the same commit as the change.

## Quality benchmark

- William considers **voronoi** and **stir** outstanding implementations:
  refined and unique. Use them as the quality benchmark for future work.
- Specifically, the boxes and panel layouts in **quantum**, **bridge**,
  **kalman** and **pendulums** feel like low-effort "AI-generated slop" to him.
  This is feedback on the visual presentation: boxed readouts alone do not
  make a design feel finished. Prioritise care, polish and distinctive
  composition when adding or improving a saver.
