# single inheritance

class Car :
    wheels = 4
    FuelType=""
    def show(self):
        print(f"wheels = {self.wheels} , fuel={self.FuelType}")
# extends car class
class EvCar(Car) :
    wheels = 4
    FuelType= "Eleectric"

# tesla = EvCar()
# tesla.show()

# multiple inheritance
class Person :
    def __init__(self,nm,age):
        self.name = nm
        self.age = age
    def show(self):
        print(f"Name={self.name} , Age={self.age}")
class Employee:
    def __init__(self,sal,exp):
        self.salary = sal
        self.experience = exp
    def show(self):
        print(f"Salary={self.salary} , Experience={self.experience}")
class EMployeePerson(Person,Employee):
    def __init__(self,nm,age,sal,exp):
        Person.__init__(self,nm,age)
        Employee.__init__(self,sal,exp)
    def show(self):
        Person.show(self)
        Employee.show(self)
# emp = EMployeePerson("John",30,50000,5)
# emp.show()


# multilevel inheritance
class Manager(EMployeePerson):
    def __init__(self,nm,age,sal,exp,team):
        EMployeePerson.__init__(self,nm,age,sal,exp)
        self.team = team
    def show(self):
        EMployeePerson.show(self)
        print(f"Team={self.team}")
# manager = Manager("Alice",35,70000,10,"Dev Team")
# manager.show()

# hierarchical inheritance
class Developer(EMployeePerson):
    def __init__(self,nm,age,sal,exp,skills):
        EMployeePerson.__init__(self,nm,age,sal,exp)
        self.skills = skills
    def show(self):
        EMployeePerson.show(self)
        print(f"Skills={self.skills}")
class SeniorManager(Manager,Developer):
    def __init__(self,nm,age,sal,exp,team,skills):
        Manager.__init__(self,nm,age,sal,exp,team)
        Developer.__init__(self,nm,age,sal,exp,skills)
    def show(self):
        print("Senior Manager Details:")
        Developer.show(self)
senior_manager = SeniorManager("Bob",40,90000,15,"QA Team","Python, Java")
senior_manager.show()