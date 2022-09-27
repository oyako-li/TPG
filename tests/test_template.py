# https://qiita.com/hoto17296/items/fa0166728177e676cd36
import unittest

class TemplateTest(unittest.TestCase):
    def test_sample(self):
        pass

# テスト結果を管理するTextTestResultを継承したクラスを作成
class VerboseTestResult(unittest.TextTestResult):
    # サブテスト毎に呼ばれるaddSubTest()をオーバーライド
    def addSubTest(self, test, subtest, outcome):
        # 元のaddSubTest()を実行して基本的な処理をさせる
        super(VerboseTestResult, self).addSubTest(test, subtest, outcome)
        # 実行数を加算する←new!
        self.testsRun += 1

# 適当な例外クラス
class HogeError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return "[{0}]: {1}".format(self.code, self.message)


# 例外を返す適当な関数
def hoge():
    raise HogeError(1234, "hoge")


class HogeTest(unittest.TestCase):
    # 発生した例外の種類だけチェックすれば十分な場合
    def test_hoge(self):
        with self.assertRaises(HogeError):
            hoge()

        # with使わない場合の書き方。with使う方がぱっと見でわかりやすいので個人的には好き。
        self.assertRaises(HogeError, hoge)

    # 発生した例外のエラーメッセージもチェックしたい場合
    def test_fuga(self):
        # エラーメッセージのチェックには正規表現が使える
        with self.assertRaisesRegex(HogeError, ".*1234.*hoge.*"):
            hoge()

    # 発生した例外の中身もチェックしたい場合
    def test_piyo(self):
        with self.assertRaises(HogeError) as cm:
            hoge()

        # コンテキストマネージャに格納された例外オブジェクト
        the_exception = cm.exception

        # 中身を好きにチェックする
        self.assertEqual(the_exception.code, 1234)


# class MainTest(unittest.TestCase):
#     def test_main(self):
#         # main関数の呼び出し自体は他の関数とかと同じ
#         # main()は標準入出力を扱う必要があるのでそこが少し面倒。
#         import sys
#         import io
#         from contextlib import redirect_stdout

#         # コマンドライン引数を擬似的に再現するためsys.argv()に自前で格納
#         # cleanしないと正しい引数を渡せないので注意
#         sys.argv.clear()
#         sys.argv.append('./sample.py')
#         sys.argv.append('--arg1')
#         sys.argv.append('hogehoge')

#         # main()の標準出力をioにリダイレクト
#         _io = io.StringIO()
#         with redirect_stdout(_io):
#             sample.main()

#         # mainの標準出力をチェック
#         self.assertEqual(_io.getvalue(), 'hogehoge\n')

import unittest.mock
import os


def print_abspath(x):
    print(os.path.abspath(x))


class MockTest(unittest.TestCase):
    def test_hoge(self):
        # モックを作成
        m = unittest.mock.MagicMock()

        # 動作を置き換えたいオブジェクトにモックを代入(パッチを当てる)
        os.path.abspath = m

        # テスト対象のコードの実行
        print_abspath('hoge')

        # モックが正しい引数で呼び出されたことのチェック
        m.assert_called_with('hoge')


if __name__ == '__main__':
    runner = unittest.TextTestRunner(resultclass = VerboseTestResult)
    unittest.main(testRunner=runner)