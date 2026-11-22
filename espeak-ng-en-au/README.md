# espeak-ng: Australian English (en-AU) voice

`wpiper/espeak-ng-data/` comes compiled with an extra voice, `en-au` ("English (Australia)"). This
folder holds the espeak-ng source changes behind it, as the GPL requires and so you can rebuild or
improve it.

Base: [espeak-ng](https://github.com/espeak-ng/espeak-ng) at commit `688cb603` (2026-11-08).

| File | What it is |
|---|---|
| `en-au.patch` | changes to `dictsource/en_rules` (Australian spelling-to-sound rules under the `?9` condition: flat TRAP in *dance*, *demand*, *sample*, ...) and `phsource/phonemes` (registers the `en-au` phoneme table) |
| `new_files/phsource/ph_english_au` | the en-AU phoneme table (vowel qualities such as `æɪ` in *mate* and `ɐʉ` in *no*) |
| `new_files/espeak-ng-data/lang/gmw/en-AU` | the voice file: `language en-au`, `phonemes en-au`, `dictrules 9`, vowel replacements |

## Rebuilding the data

1. Clone espeak-ng, check out `688cb603`, apply the patch and copy in the new files:
   ```
   git apply path\to\en-au.patch
   copy new_files\phsource\ph_english_au phsource\
   copy new_files\espeak-ng-data\lang\gmw\en-AU espeak-ng-data\lang\gmw\
   ```
2. Build espeak-ng (see its `docs/building.md`). Then compile the phonemes and the English dictionary:
   ```
   espeak-ng --compile-phonemes
   espeak-ng --compile=en
   ```
3. Copy these files into `wpiper/espeak-ng-data/`: `phondata`, `phondata-manifest`, `phonindex`,
   `phontab`, `en_dict` and `lang/gmw/en-AU`.

Only these files differ from the stock data. The other languages are unchanged.

Check the result:

```
python -c "from wpiper.phonemize_espeak import EspeakPhonemizer as E; print(E().phonemize('en-au', 'G\'day mate, no worries.'))"
```
