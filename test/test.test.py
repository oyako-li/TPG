import unittest

class MyTestCase(unittest.TestCase):
  def test_test3(self):
    self.assertIn("hoge", "hogehoge") # これはOK

  def test_test4(self):
    self.assertIn("hoge", ["hoge", "fuga"]) # これもOK

  def test_test5(self):
    # self.assertIn(["hoge", "fuga"], ["hoge", "fuga", "fuga"]) # これはNG
    # self.assertIn(["hoge", "fuga"], ["hoge", "fuga"]) # これでもNG
    pass

  def test_test6(self):
    self.assertIn(["hoge", "fuga"], [["hoge", "fuga"] ,["hoge", "hoge", "fuga", "fuga"]]) # これはOK

  def test_test7(self):
    self.assertIn(("hoge", "fuga"), (("hoge", "fuga"), ("hoge", "hoge", "fuga", "fuga"))) # これもOK

def test_div(self):
    with self.assertRaises(ZeroDivisionError):
      return (5 / 0) # これはテストOK
      
if __name__ == '__main__':
    unittest.main()