# 1M eos pad 

Generated on: 03052025164544
Using: $ python prepare_var_len.py --config ./recipie/config-eospad.json --train 1000000 --test 1000

Total Checked Samples: 2361668

== Train ==
Valid (target): 1000000 (1000000)
Text Tokens: 768,424,740
Tot Tokens: 1,024,000,000
Average Text Tokens per Sample: 768.42

== Test ==
Valid (target): 1000 (1000)
Text Tokens: 768,424,740
Tot Tokens: 1,024,000
Average Text Tokens per Sample: 805.65

== Hyper-params ==
EOS token id: 50256
PAD token id: 50257
Format: [text tokens] [50256] [50257, ..., 50257]

# 1M eos eos 

Metadata for Test / Train Datasets 03052025175151

Generated on: 03052025175151
Using: $ python prepare_var_len.py --config ./recipie/config-eospad.json --train 1000000 --test 1000

Total Checked Samples: 2361668

== Train ==
Valid (target): 1000000 (1000000)
Text Tokens: 768,424,740
Tot Tokens: 1,024,000,000
Average Text Tokens per Sample: 768.42

== Test ==
Valid (target): 1000 (1000)
Text Tokens: 768,424,740
Tot Tokens: 1,024,000
Average Text Tokens per Sample: 805.65

== Hyper-params ==
EOS token id: 50256
PAD token id: 50256
Format: [text tokens] [50256] [50256, ..., 50256]