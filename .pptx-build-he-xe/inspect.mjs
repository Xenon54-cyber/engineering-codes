import fs from "node:fs/promises";
import { installNode } from "@oai/granola/node";
import { Presentation } from "@oai/artifact-tool";

installNode();
const deck = await Presentation.load("source.pptx");
const details = await deck.inspect({
  select: "$.slides[*]",
  include: ["id", "index", "layout", "title", "elements.id", "elements.kind", "elements.text", "elements.position", "elements.style", "elements.paragraphs", "notes.text"],
});
await fs.writeFile("inspect.ndjson", details.ndjson, "utf8");
const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile("montage-before.webp", new Uint8Array(await montage.arrayBuffer()));
for (const [i, slide] of deck.slides.items.entries()) {
  const png = await slide.export({ format: "png", scale: 1 });
  await fs.writeFile(`slide-${String(i + 1).padStart(2, "0")}.png`, new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({ slides: deck.slides.items.length, inspect: "inspect.ndjson" }));
