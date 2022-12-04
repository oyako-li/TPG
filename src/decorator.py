def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")

def decorator(func):
    def wrapper(*args, **kwargs):
        greet = func(*args, **kwargs)
        return greet+'world'

    return wrapper

class Sample:
    def __init__(self) -> None:
        self.sample = 1
        self.test = 22

    @decorator
    def decoratoee(self, _sample='hello', _ex='./e'):
        return _sample+_ex+str(self.sample)

if __name__ == '__main__':
    say_whee()
    sa = Sample()
    print(sa.decoratoee(_sample='change'))