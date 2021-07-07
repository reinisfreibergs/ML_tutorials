class Animal(object):
    def __init__(self):
        self.__hunger_perc = 0.7

    def get_hunger_perc(self):
        return self.__hunger_perc

    def eat(self):
        self.__hunger_perc -= 0.1
        self.__hunger_perc = max(0, self.__hunger_perc)

    def sleep(self, hours):
        self.__hunger_perc += 0.1 * hours
        self.__hunger_perc = min(self.__hunger_perc, 1)

    def move(self):
        pass


class Dog(Animal):
    def __init__(self):
        super().__init__()
        self.__bones_hidden = 0

    def move(self):
        self.__hunger_perc += 0.1
        self.__hunger_perc = max(0, self.__hunger_perc)

    def bark(self):
        pass


class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.__items_destroyed = 0

    def move(self):
        self.__hunger_perc += 1e-2
        self.__hunger_perc = max(0, self.__hunger_perc)

    def meow(self):
        print('meow')


class Robot(object):
    def __init__(self):
        super().__init__()
        self.__battery_perc = 1.0

    def move(self):
        self.__battery_perc -= 1e-1
        self.__battery_perc = max(0, self.__battery_perc)

    def charge(self, hours):
        self.__battery_perc += 0.1 * hours
        self.__battery_perc = min(self.__battery_perc, 1)
