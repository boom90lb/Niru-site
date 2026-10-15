/**
 * The training step as one term, for either model: the model and its loss,
 * a gradient derived from them, and an update. A model with state outside
 * the gradient, such as DeepSeek's gate bias, also draws the rule that
 * updates it.
 */
import { Sketch } from "./kit.ts";
import type { View } from "./view.ts";

export function trainingStep(o: {
  id: string;
  model: string;
  note: string;
  theta: string;
  loss: string;
  update: string;
  caption: string;
  state?: { name: string; rule: string };
}): View {
  const s = new Sketch();
  for (const [y, name] of [
    [0.6, "inputs"],
    [-0.6, "labels"],
  ] as const) {
    s.wire(
      [
        [-7, y],
        [0.8, y],
      ],
      "ink",
    );
    s.arrows(-7, 0.8, y, "ink");
    s.text(-7, y, name, { italic: true, anchor: "end", dx: -6, dy: 4 });
  }
  const model = s.slab(4, 0, 6.4, 4.6, { chip: true, label: "model", note: o.note });
  s.open("model", "the model", o.model, model);
  s.wire(
    [
      [7.2, 0],
      [12.8, 0],
    ],
    "ink",
  );
  s.text(10, 0, o.loss, { italic: true, anchor: "middle", dy: -6 });
  s.slab(16, 0, 6.4, 4.6, { dashed: true, label: "grad", note: "derived, never stored" });
  s.wire(
    [
      [19.2, 0],
      [22.6, 0],
      [26.4, -3],
      [26.8, -3],
    ],
    "ink",
  );
  s.text(22.4, 0, "g", { italic: true, dy: -6 });
  s.slab(30, -3, 6.4, 4.6, { label: "update", note: o.update });
  // Parameters: read by the model, by the derivative and by the update.
  s.wire(
    [
      [-7, -8],
      [30, -8],
    ],
    "ochre",
    { w: 2.2 },
  );
  s.text(-7, -8, o.theta, { italic: true, ink: "ochre", anchor: "end", dx: -6, dy: 4 });
  for (const x of [4, 16]) {
    s.wire(
      [
        [x, -8],
        [x, -2.3],
      ],
      "ochre",
      { w: 2.2 },
    );
    s.copy(x, [-8], ["ochre"]);
  }
  s.wire(
    [
      [30, -8],
      [30, -5.3],
    ],
    "ochre",
    { w: 2.2 },
  );
  s.wire(
    [
      [33.2, -3],
      [40, -3],
    ],
    "ochre",
    { w: 2.2 },
  );
  s.text(40, -3, "θ′, for the next step", { italic: true, ink: "ochre", dx: 6, dy: 4 });
  // The optimizer's state, carried from step to step.
  const loop: Array<[number, number, number]> = [];
  for (let i = 0; i <= 24; i++) {
    const a = Math.PI * (i / 24);
    loop.push([30 + Math.cos(a) * 2, -0.7 + Math.sin(a) * 1.9, 0]);
  }
  s.add({ kind: "stroke", pts: loop, w: 1.3, ink: "faint", layer: 0 });
  s.text(30, 1.2, "moments m, v and a step counter", {
    italic: true,
    size: 11,
    ink: "faint",
    anchor: "middle",
    dy: -6,
  });
  if (o.state) {
    // State outside the gradient: the model reads it, and a rule updates it from what the model counted.
    const y = 7.4;
    s.wire(
      [
        [-7, y],
        [21, y],
      ],
      "faint",
      { dash: [4, 3] },
    );
    s.text(-7, y, o.state.name, {
      italic: true,
      size: 11,
      ink: "faint",
      anchor: "end",
      dx: -6,
      dy: 4,
    });
    s.copy(2.6, [y], ["faint"]);
    s.wire(
      [
        [2.6, y],
        [2.6, 2.3],
      ],
      "faint",
      { dash: [4, 3] },
    );
    s.wire(
      [
        [5.6, 2.3],
        [5.6, 4.9],
        [21, 4.9],
      ],
      "ink",
    );
    s.text(8, 4.9, "expert counts c", { italic: true, size: 11, dy: -6 });
    s.slab(25, 6.15, 8, 3.6, { label: "bias rule", size: 13, note: o.state.rule });
    s.wire(
      [
        [29, 6.15],
        [40, 6.15],
      ],
      "faint",
      { dash: [4, 3] },
    );
    s.text(40, 6.15, "b′, for the next step", {
      italic: true,
      size: 11,
      ink: "faint",
      dx: 6,
      dy: 4,
    });
  }
  return {
    id: o.id,
    title: "The training step",
    caption: o.caption,
    sketch: s,
    yaw: -0.2,
    pitch: 0.3,
  };
}
