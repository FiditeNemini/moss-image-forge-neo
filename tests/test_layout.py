# MOSS-Image support for Forge Neo
# The layout prompt with the braces sd-dynamic-prompts strips put back, character for character.

import json
import unittest

from forge_moss.native.layout import restore_layout_json


def strip(text: str) -> str:
    # what sd-dynamic-prompts leaves of a layout: every {x} read as a one-option variant, braces gone
    return text.replace("{", "").replace("}", "")


# what Regional Boxes writes (OpenMOSS's compiler style): compact, on one line
COMPACT = (
    '{"summary":{"meta":["poster","flat illustration"],"canvas":"single","caption":"A minimalist winter travel poster '
    'with a pale blue background, a red fox sitting in fresh snow, a bold dark-navy title at the top and a small '
    'subtitle at the bottom. Flat vector illustration."},"items":[{"label":"text","caption":"Large bold dark-navy '
    'uppercase title, centered","text":"WINTER FOX","bbox":[-528,-352,-368,352]},{"label":"object","caption":"A red '
    'fox sitting in fresh snow, flat vector style","bbox":[-276,-317,422,314]},{"label":"text","caption":"Small thin '
    'navy subtitle, centered","text":"FORGE NEO","bbox":[472,-235,562,232]}],"bbox_coord_space":"abs_centralized",'
    '"bbox_format":"yxyx"}'
)

# OpenMOSS's user guide example, as written there: indented, with spaces
GUIDE = """{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."
  },
  "items": [
    {
      "label": "text",
      "caption": "Large dark-blue uppercase title centered at the top.",
      "text": "EXPLORE",
      "bbox": [-416, -384, -288, 384]
    },
    {
      "label": "object",
      "caption": "A flat blue mountain illustration in the lower center.",
      "bbox": [-192, -288, 384, 288]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}"""


class TestRestoreLayout(unittest.TestCase):
    def test_compact_layout_comes_back_exactly(self):
        self.assertEqual(restore_layout_json(strip(COMPACT)), COMPACT)

    def test_guide_example_comes_back(self):
        # indented: the same layout, though a brace may land on the other side of the line break it stood by
        restored = restore_layout_json(strip(GUIDE))
        self.assertEqual(json.loads(restored), json.loads(GUIDE))
        self.assertEqual(strip(restored), strip(GUIDE))

    def test_the_prompt_dynamic_prompts_gave_the_model(self):
        # the prompt of the Forge Neo infotext, from the panel with sd-dynamic-prompts on
        seen = ('"summary":"meta":["poster","flat illustration"],"canvas":"single","caption":"A poster.","items":['
                '"label":"text","caption":"Title","text":"WINTER FOX","bbox":[-528,-352,-368,352],"label":"object",'
                '"caption":"A fox","bbox":[-276,-317,422,314]],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"')
        layout = json.loads(restore_layout_json(seen))
        self.assertEqual(layout["summary"], {"meta": ["poster", "flat illustration"], "canvas": "single",
                                             "caption": "A poster."})
        self.assertEqual([item["label"] for item in layout["items"]], ["text", "object"])
        self.assertEqual(layout["items"][1], {"label": "object", "caption": "A fox", "bbox": [-276, -317, 422, 314]})

    def test_strings_with_json_characters_and_line_breaks(self):
        layout = {"summary": {"caption": 'A sign: "OPEN, 24h" [neon], tags: a, b'},
                  "items": [{"label": "text", "caption": "Two lines, left aligned", "text": "OPEN\n24h: [ok]",
                             "bbox": [0, 0, 10, 10]},
                            {"label": "character", "caption": "A clerk", "tags": ["smiling", "apron"],
                             "palette": ["red"], "bbox": [10, 10, 20, 20]}],
                  "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
        for text in (json.dumps(layout), json.dumps(layout, ensure_ascii=False, separators=(",", ":"))):
            self.assertEqual(restore_layout_json(strip(text)), text)

    def test_items_whose_keys_come_in_another_order(self):
        layout = ('{"summary":{"caption":"x"},"items":[{"bbox":[0,0,1,1],"label":"object","caption":"a"},'
                  '{"bbox":[1,1,2,2],"label":"object","caption":"b"}],"bbox_coord_space":"abs_centralized",'
                  '"bbox_format":"yxyx"}')
        self.assertEqual(restore_layout_json(strip(layout)), layout)

    def test_chinese_text_and_empty_items(self):
        layout = json.dumps({"summary": {"caption": "冰箱上的便利贴"}, "items": [], "bbox_coord_space": "abs_centralized",
                             "bbox_format": "yxyx"}, ensure_ascii=False)
        self.assertEqual(restore_layout_json(strip(layout)), layout)

    def test_other_prompts_are_left_alone(self):
        for text in (COMPACT, GUIDE, "A cat on a sofa.", "", '"summary": a cat', '"summary":"caption":"x"',
                     '"items":["label":"text"', '"summary":"caption":"x","items":[] and more',
                     '"summary":"caption":"x","items":[],"bbox_format":"yxyx" trailing'):
            self.assertEqual(restore_layout_json(text), text, text)


if __name__ == "__main__":
    unittest.main()
