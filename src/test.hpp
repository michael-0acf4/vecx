#pragma once

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static const std::string red(const std::string &s) {
  return "\033[31m" + s + "\033[0m";
};
static const std::string green(const std::string &s) {
  return "\033[32m" + s + "\033[0m";
};
static const std::string yellow(const std::string &s) {
  return "\033[33m" + s + "\033[0m";
};

#define TEST(test_name)                                                        \
  static std::string test_name(); /* declaration */                            \
  static bool _##test_name =                                                   \
      add_test(#test_name, test_name); /* static register */                   \
  static std::string test_name()       /* definition */

#define ASSERT(expr)                                                           \
  if (!(expr)) {                                                               \
    std::ostringstream oss;                                                    \
    oss << red("  Assertion failed: ") << "" #expr " at " << __FILE__ << ":"   \
        << __LINE__ << "\n";                                                   \
    return oss.str();                                                          \
  }

#define _EPSILON 10e-6
#define ASSERT_CLOSE(left, right, precision)                                   \
  if (!(fabs(left - right) <= precision)) {                                    \
    std::ostringstream oss;                                                    \
    std::string sleft =                                                        \
        "\n   " + yellow(#left) + " (" + std::to_string(left) + ")";           \
    std::string sright =                                                       \
        " vs " + yellow(#right) + " (" + std::to_string(right) + ")";          \
                                                                               \
    oss << red("  Assertion failed (close): ") << sleft << sright << "\n  at " \
        << __FILE__ << ":" << __LINE__ << "\n";                                \
    return oss.str();                                                          \
  }

#define DEBUG(where, expr)                                                     \
  std::cout << yellow(#where) << yellow(" debug :: ") << #expr << " = "        \
            << expr << "\n";
#define DEBUG_NUMBER(where, expr)                                              \
  std::cout << yellow(#where) << yellow(" debug :: ") << std::setprecision(30) \
            << #expr << " = " << expr << "\n";
#define LGTM return "";

static std::vector<std::pair<const char *, std::function<std::string()>>> &
all_tests() {
  static std::vector<std::pair<const char *, std::function<std::string()>>> t;
  return t;
}
static bool add_test(const char *name, std::function<std::string()> fn) {
  all_tests().emplace_back(name, fn);
  return true;
}

int run_all_tests() {
  int number = 1;
  int sucess = 0;
  int fail = 0;
  auto tests = all_tests();
  size_t total_tests = tests.size();
  std::cout << "Found " << total_tests << " tests.."
            << "\n----------\n";
  for (const auto &test : tests) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto err = test.second();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    if (err.empty()) {
      sucess++;
    } else {
      fail++;
    }

    const auto status = (err.empty() ? green("OK") : red("FAIL"));
    std::cout << "[TEST" << std::setw(2) << (number++) << "] " << test.first
              << ": " << status << " (" << ms_int.count() << " ms)\n"
              << err;
  }

  if (fail > 0) {
    std::cout << "\n----------\n"
              << red(std::to_string(fail)) + red("/") +
                     red(std::to_string(total_tests))
              << red(" Failed") << "\n";
    return -1;
  } else {
    auto tot_str = green(std::to_string(total_tests));
    std::cout << "\n----------\n"
              << tot_str + green("/") + tot_str + green(" Succeded") << "\n";
    return 0;
  }
}
