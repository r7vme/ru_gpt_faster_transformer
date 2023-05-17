#include "bpe.h"

RE2 re(
    "('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)");

void test_re2() {
  assert(re.ok());  // compiled; if not, see re.error();

  std::string w;
  std::string text = "we'd annoyingly 顽皮";
  re2::StringPiece input(text);

  std::vector<std::string> v;
  while (RE2::FindAndConsume(&input, re, &w)) {
    v.push_back(w);
  }
  assert(v == std::vector<std::string>({"we", "\'d", " annoyingly", " 顽皮"}));
}

void test_bytes_to_unicode() {
  std::unordered_map<uint8_t, wchar_t> b2u;
  std::unordered_map<wchar_t, uint8_t> u2b;
  bytes_to_unicode(&b2u, &u2b);
  assert(b2u.size() == 256);
  assert(b2u[0] == 0x100);
  assert(u2b.size() == 256);
}

void test_byte_encode_token() {
  std::unordered_map<uint8_t, wchar_t> b2u;
  bytes_to_unicode(&b2u, NULL);

  std::string s(" very");
  std::wstring b;
  byte_encode_token(s, b2u, &b);
  assert("Ġvery" == wstring_to_utf8(b));
}

void test_load_merge_rules() {
  BPERanks bpe_ranks;
  std::fstream merges("/tmp/merges.txt", std::ios::in);
  load_merge_rules(merges, &bpe_ranks);
  assert(bpe_ranks.size() == 50000);

  auto iter = bpe_ranks.find({utf8_to_wstring("Ġg"), utf8_to_wstring("azed")});
  assert(iter != bpe_ranks.end());
  assert(iter->second = 49999);
}

void test_get_pairs() {
  std::vector<std::pair<std::wstring, std::wstring> > pairs;
  get_pairs(utf8_to_wstring("very"), &pairs);
  assert(pairs.size() == 3);
  assert(wstring_to_utf8(pairs[1].first) == "e");
  assert(wstring_to_utf8(pairs[1].second) == "r");
}

void test_bpe() {
  BPERanks bpe_ranks;
  std::fstream merges("/tmp/merges.txt", std::ios::in);
  load_merge_rules(merges, &bpe_ranks);
  assert(bpe_ranks.size() == 50000);

  std::vector<std::wstring> result;
  bpe(utf8_to_wstring("annoyingly"), bpe_ranks, &result);
  assert(result == std::vector<std::wstring>({L"ann", L"oy", L"ingly"}));

  result.clear();
  bpe(utf8_to_wstring("very"), bpe_ranks, &result);
  assert(result == std::vector<std::wstring>({L"very"}));
}

auto _print_string_vec = [](std::vector<std::string>& v) {
  // To be compatible with Python's print(*lst, sep=', ')
  for (int i = 0; i < v.size(); ++i) {
    std::cout << v[i];
    if (i < v.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
};

void test_tokenize() {
  assert(re.ok());  // compiled; if not, see re.error();

  BPERanks bpe_ranks;
  std::fstream merges("/tmp/merges.txt", std::ios::in);
  load_merge_rules(merges, &bpe_ranks);

  std::unordered_map<uint8_t, wchar_t> b2u;
  bytes_to_unicode(&b2u, NULL);

  std::vector<std::string> candidates = {
      "this is <|endoftext|> else<|endoftext|>",
      "<|endoftext|> else<|endoftext|>", "this is <|endoftext|> else",
      "this is <|endoftext|>else", "this is else"};
  for (auto s : candidates) {
    std::vector<std::string> result;
    tokenize(s, re, bpe_ranks, b2u, &result);
    _print_string_vec(result);
  }
}

void test_tokenize_regression() {
  assert(re.ok());  // compiled; if not, see re.error();

  BPERanks bpe_ranks;
  std::fstream merges("/tmp/merges.txt", std::ios::in);
  load_merge_rules(merges, &bpe_ranks);

  std::unordered_map<uint8_t, wchar_t> b2u;
  bytes_to_unicode(&b2u, NULL);

  // In order to make sure that /tmp/sample.txt contains many lines of
  // text of different langauges, I download the lyrics data from
  // https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres,
  // and ran the following commands to randomly sample 10000 lines.
  /*
     cat /tmp/lyrics-data.csv | head -n 1000000  | awk 'NF' | sort | \
     uniq | sort -R | head -n 10000 > /tmp/sample.txt
  */
  std::fstream ins("/tmp/sample.txt", std::ios::in);
  std::string line;
  while (std::getline(ins, line)) {
    std::vector<std::string> result;
    tokenize(line, re, bpe_ranks, b2u, &result);
    _print_string_vec(result);
  }
}

void test_load_vocab() {
  std::unordered_map<std::string, int> t2i;
  std::unordered_map<int, std::string> i2t;
  std::fstream vocab_txt("/tmp/vocab.txt", std::ios::in);
  load_vocab(vocab_txt, &t2i, &i2t);
  assert(t2i.size() == 50257);
  assert(i2t.size() == 50257);
  assert(t2i["\""] == 1);
  assert(i2t[1] == "\"");
}

void test_encode_decode() {
  BPERanks bpe_ranks;
  std::fstream merges("/tmp/merges.txt", std::ios::in);
  load_merge_rules(merges, &bpe_ranks);

  std::unordered_map<uint8_t, wchar_t> b2u;
  std::unordered_map<wchar_t, uint8_t> u2b;
  bytes_to_unicode(&b2u, &u2b);

  std::unordered_map<std::string, int> t2i;
  std::unordered_map<int, std::string> i2t;
  std::fstream vocab_txt("/tmp/vocab.txt", std::ios::in);
  load_vocab(vocab_txt, &t2i, &i2t);

  std::vector<std::string> candidates = {
      "this is <|endoftext|> else<|endoftext|>",
      "<|endoftext|> else<|endoftext|>", "this is <|endoftext|> else",
      "this is <|endoftext|>else", "this is else"};
  for (auto s : candidates) {
    std::vector<int> ids;
    encode(s, re, bpe_ranks, b2u, t2i, &ids);
    assert(ids.size() > 0);
    assert(decode(ids, u2b, i2t) == s);
  }
}

int main() {
  test_bytes_to_unicode();
  test_re2();
  test_load_merge_rules();
  test_byte_encode_token();
  test_get_pairs();
  test_bpe();
  test_tokenize();
  test_load_vocab();
  test_encode_decode();
  return 0;
}
