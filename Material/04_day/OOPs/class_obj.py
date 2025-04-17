class Car :
    # class variables 
    # name = "xyz"
    # year = 2020
    def __init__(self,nm,yr):
        # instance variable
        self.name = nm
        self.year = yr
    #=methods 
    def display(self):
        print(f"car name is {self.name} , year {self.year}")
    
mycar =Car("honda city",2050)
mycar.display()