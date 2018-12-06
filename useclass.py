from student import Student, PakistaniStudent

student1 = Student("Samir", "CS", 4.0, False)
student2 = PakistaniStudent("Raja Pakistani", "CS", 3.1, False)


print(student1.gpa)

print(student1.is_pakistani())
print(student2.is_pakistani())