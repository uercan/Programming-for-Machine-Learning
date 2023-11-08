import turtle
##Exercise 14.1
def sed(string1, string2, file1, newfile):
    f1 = open(file1, "r")
    f2 = open(newfile, "w")
    text = f1.read()
    newtext = text.replace(string1, string2)
    f2.write(newtext)
    f1.close()
    f2.close()


str1 = 'yabadubidubi'
str2 = 'aiaiaiiaiai'
originalfile = 'testing.txt'
replace = originalfile + '.replaced'

sed(str1, str2, originalfile, replace)
f1 = open(originalfile, 'r')
f2 = open(replace, 'r')
print("\noriginal file: \n", f1.read(), "\nreplaced file: \n", f2.read())


##Task 15.2

class rectangle:
    __sides = 4

    def __init__(self, length, width):
        self.length = length
        self.width = width
        if self.length >= 1:
            self.side1 = self.length
            self.side3 = self.length
        else:
            self.side1 = 1
            self.side3 = 1
        if self.width >= 1:
            self.side2 = self.width
            self.side4 = self.width
        else:
            self.side2 = 1
            self.side4 = 1


def draw_rect(turt, rect):
    turt.fd(rect.side1)
    turt.rt(90)
    turt.fd(rect.side2)
    turt.rt(90)
    turt.fd(rect.side3)
    turt.rt(90)
    turt.fd(rect.side4)


def draw_circle(turt, radius):
    turt.circle(radius)
    turtle.mainloop()


rect1 = rectangle(300, 200)
bob = turtle

draw_rect(bob, rect1)
draw_circle(bob, 50)


##Task 17.2

class Kangaroo:
    def __init__(self, name, contents = None):
        self.name = name
        if contents == None:
            contents = []
        self.pouch_contents = contents

    def put_in_pouch(self, anyobject):
        self.pouch_contents.append(anyobject)

    def __str__(self):
        array = [ self.name + ' has the following in the pouch:' ]
        for obj in self.pouch_contents:
            s = '\n' + object.__str__(obj)
            array.append(s)
        return '\n'.join(array)

kanga = Kangaroo('Kanga')
roo = Kangaroo('roo')
kanga.put_in_pouch(["yes","no"])
kanga.put_in_pouch(roo)
print(kanga)
print(roo)

##TASK 18.1 UML DIAGRAM, it is done in a separate pdf file
##which is called TASK 18.1 UML DIAGRAM

