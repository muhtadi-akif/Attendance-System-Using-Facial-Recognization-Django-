
from django.shortcuts import render, redirect

from records.models import *

import cv2
import numpy as np

import os

# from settings import BASE_DIR
# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

course = Course.objects.all()
teachers = Teachers.objects.all()
semester = Semester.objects.all()
grades = Grades.objects.all()
class_info = Class_info.objects.all()
roles = Roles.objects.all()
department = Department.objects.all()
post = Post.objects.all()
timing = Timing.objects.all()
room = Rooms.objects.all()
students = Students.objects.all()
teacher_courses = Teacher_courses.objects.all()
student_courses = Student_courses.objects.all()

context = {
    'course': course,
    'semester': semester,
    'grades': grades,
    'class_info': class_info,
    'roles': roles,
    'department': department,
    'post': post,
    'timing': timing,
    'room': room,
    'teacher_courses': teacher_courses,
}


# def details(request, id):
#     students = Students.objects.get(id=id)
#     context = {
#         'students' : students
#     }
#     return render(request, 'studentInfo.html', context)


def index(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            students = Students.objects.get(institute_id=username, password=1234)
            request.session['student_id'] = students.id
            # return render(request,'students.html', {'student_id': students})
            return render(request, 'students.html', {'student': students})

        except Students.DoesNotExist:
            try:
                teachers = Teachers.objects.get(initial=username, password=password)
                request.session['teacher_id'] = teachers.id
                print(teachers.name)
                return render(request, 'teachers.html', {'teacher': teachers})


            except Teachers.DoesNotExist:
                admins = Admins.objects.get(name=username, password=password)
                request.session['admin_id'] = admins.id
                return render(request, 'adminmain.html', {'admin': admins})

    return render(request, 'home.html')


def studentlogout(request):
    try:
        del request.session['student_id']

    except KeyError:
        pass
    return redirect("/")


def teacherlogout(request):
    try:
        del request.session['teacher_id']

    except KeyError:
        pass
    return redirect('/')


def adminlogout(request):
    try:
        del request.session['admin_id']

    except KeyError:
        pass
    return redirect('/')


def adminmain(request):
    if 'admin_id' not in request.session:
        return redirect(index)
    else:
        admin_id = request.session['admin_id']
        admins = Admins.objects.get(id=admin_id)
        return render(request, 'adminmain.html', {'admin': admins})


def adminstudent(request):
    if 'admin_id' not in request.session:
        return redirect(index)
    else:
        admin_id = request.session['admin_id']
        admins = Admins.objects.get(id=admin_id)

        if request.method == 'POST':
            if request.POST.get('id') != 0:
                post = Students()

                post.institute_id = request.POST.get('institute_id')
                post.password = request.POST.get('password')
                post.confirm_password = request.POST.get('confirm_password')
                post.name = request.POST.get('name')
                post.address = request.POST.get('address')
                post.phone_number = request.POST.get('phone')
                post.email_address = request.POST.get('email')
                post.role_id = Roles.objects.get(id=request.POST.get('role_id'))

                post.dept_id = Department.objects.get(id=request.POST.get('dept_id'))
                post.save()
                # role.save()
                # dept.save()
                return render(request, 'adminstudent.html', context)
        else:  # 'id,institute_id, password, name, address, phone, email, role'
            return render(request, 'adminstudent.html', context)


def adminteacher(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Teachers()

            post.id = request.POST.get('id')
            post.initial = request.POST.get('initial')
            post.password = request.POST.get('password')
            post.name = request.POST.get('name')
            post.dept_id = Department.objects.get(id=request.POST.get('dept_id'))
            post.post_id = Post.objects.get(id=request.POST.get('post_id'))
            post.role_id = Roles.objects.get(id=request.POST.get('role_id'))
            post.office_location = request.POST.get('office_location')

            post.save()
            # role.save()
            # dept.save()
            return render(request, 'adminteacher.html', context)
    return render(request, 'adminteacher.html', context)


def admindepartment(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Department()

            post.id = request.POST.get('id')
            post.name = request.POST.get('name')
            post.location = request.POST.get('location')

            post.save()
            return render(request, 'admindepartment.html', context)
    return render(request, 'admindepartment.html', context)


def admincourses(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Course()

            post.id = request.POST.get('id')
            post.name = request.POST.get('name')
            post.title = request.POST.get('title')
            post.description = request.POST.get('description')
            post.credit = request.POST.get('credit')

            post.save()
            return render(request, 'admincourses.html', context)
    return render(request, 'admincourses.html', context)


def adminpost(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Post()

            post.name = request.POST.get('name')
            post.save()
            return render(request, 'adminpost.html')
    return render(request, 'adminpost.html')


def adminroom(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Rooms()

            post.building = request.POST.get('building')
            post.room_no = request.POST.get('room_no')

            post.save()
            return render(request, 'adminroom.html')
    return render(request, 'adminroom.html')


def adminsemester(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Semester()

            post.id = request.POST.get('id')
            post.name = request.POST.get('name')
            post.year = request.POST.get('year')
            post.start_date = request.POST.get('start_date')
            post.end_date = request.POST.get('end_date')

            post.save()
            return render(request, 'adminsemester.html')
    return render(request, 'adminsemester.html')


def adminclassinfo(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            # post = Class_info()

            class_info = Class_info(timing_id=request.POST.get(id='time_id'),
                                    semester_id=request.POST.get(id='semester_id'),
                                    rooms_id=request.POST.get(id='room_id'))

            class_info.save()
            return render(request, 'adminclassinfo.html', context)
    return render(request, 'adminclassinfo.html', context)


def admingrade(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Grades()

            post.id = request.POST.get('id')
            post.point = request.POST.get('point')
            post.name = request.POST.get('name')

            post.save()
            return render(request, 'admingrade.html')
    return render(request, 'admingrade.html')


def admintiming(request):
    if request.method == 'POST':
        if request.POST.get('id') != 0:
            post = Timing()

            post.days = request.POST.get('days')
            post.start_time = request.POST.get('start_time')
            post.end_time = request.POST.get('end_time')

            post.save()
            return render(request, 'admintiming.html', context)
    return render(request, 'admintiming.html', context)


def teacheraddcourse(request):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)

        if request.method == 'POST':
            if request.POST.get('id') != 0:
                class_info = Class_info(timing_id=Timing.objects.get(id=request.POST.get('time_id')),
                                        semester_id=Semester.objects.get(id=request.POST.get('semester_id')),
                                        rooms_id=Rooms.objects.get(id=request.POST.get('room_id')))

                # postone.timing_id_id = request.POST.get('time_id')
                # postone.semester_id_id = request.POST.get('semester_id')
                # postone.rooms_id_id = request.POST.get('room_id')
                # postone.save()

                class_info.save()

                return render(request, 'teacheraddcourse.html', {'teacher': teachers, 'timing': timing,
                                                                 'semester': semester, 'room': room,
                                                                 'course': course, 'class_info': class_info})

    return render(request, 'teacheraddcourse.html', {'teacher': teachers, 'timing': timing, 'semester': semester,
                                                     'room': room, 'course': course})


def studentaddcourse(request):
    if 'student_id' not in request.session:
        return redirect(index)
    else:
        student_id = request.session['student_id']
        students = Students.objects.get(id=student_id)

        students_courses = Student_courses.objects.all()
        notsubmittedgrade = Grades.objects.latest('id')

        # teacher_course_new = Teacher_courses.objects.get(id = 3)

        # course = Course.objects.filter(id=teacher_course_new.course_id)

        if request.method == 'POST':
            if request.POST.get('id') != 0:
                students_courses = Student_courses(student_id=Students.objects.get(id=request.POST.get('student_id')),
                                                   teacher_course_id=Teacher_courses.objects.get(
                                                       id=request.POST.get('teacher_courses_id')),
                                                   grade_id=Grades.objects.get(id=request.POST.get('grade_id')))
                students_courses.save()

                return render(request, 'studentaddcourse.html',
                              {'student': students, 'teacher_courses': teacher_courses,
                               'teacher': teachers, 'timing': timing, 'semester': semester,
                               'room': room, 'course': course, 'class_info': class_info,
                               'notsubmittedgrade': notsubmittedgrade, 'students_courses': students_courses,
                               'grades': grades})

        return render(request, 'studentaddcourse.html', {'student': students,
                                                         'teacher': teachers, 'timing': timing, 'semester':
                                                             semester, 'teacher_courses': teacher_courses,
                                                         'grades': grades,
                                                         'notsubmittedgrade': notsubmittedgrade,
                                                         'room': room, 'course': course, 'class_info': class_info})


def studentlist(request):
    if 'admin_id' not in request.session:
        return redirect(index)
    else:
        admin_id = request.session['admin_id']
        admins = Admins.objects.get(id=admin_id)

    students = Students.objects.all()
    context = {
        'student': students,
        'admin': admins
    }

    return render(request, 'studentlist.html', context)


def teacherlist(request):
    if 'admin_id' not in request.session:
        return redirect(index)
    else:
        admin_id = request.session['admin_id']
        admins = Admins.objects.get(id=admin_id)

    teachers = Teachers.objects.all()
    context = {'teacher': teachers, 'admin': admins}

    return render(request, 'teacherlist.html', context)


def takeattandance(request):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)
        teacher_courses = Teacher_courses.objects.filter(teacher_id=teachers.id)

        context = {

            'teacher_courses': teacher_courses,
            'teacher': teachers,

        }

        return render(request, 'takeeattandance.html', context)


def coursestudentlist(request, id):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)

        student_course = Student_courses.objects.filter(teacher_course_id=id)
        teacher_courses = Teacher_courses.objects.get(id=id)

        context = {

            'teacher': teachers,
            'student_courses': student_course,
            'teacher_course_id': id,
            'teacher_courses': teacher_courses,


        }

        return render(request, 'coursestudentlist.html', context)


def attandancelist(request):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)

        return render(request, 'attandancelist.html')


def enrolledcourses(request):
    if 'student_id' not in request.session:
        return redirect(index)
    else:
        student_id = request.session['student_id']
        students = Students.objects.get(id=student_id)

        student_courses = Student_courses.objects.filter(student_id = students.id)


        context={
            'student_courses':student_courses,
            'student':students
        }


        return render(request, 'enrolledcourses.html', context)




def teacheraddcourse_two(request):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)
        print('yaay')
        postclaainfo = Class_info.objects.latest('id')

        if request.method == 'POST':
            if request.POST.get('id') != 0:
                teacher_courses = Teacher_courses(course_id=Course.objects.get(id=request.POST.get('course_id')),
                                                  teacher_id=Teachers.objects.get(id=request.POST.get('teacher_id')),
                                                  class_info_id=Class_info.objects.get(
                                                      id=request.POST.get('class_info')),
                                                  status=0, class_count=0)

                teacher_courses.save()

                print('4')

                return render(request, 'teacheraddcourse_two.html',
                              {'teacher': teachers, 'timing': timing, 'semester': semester,
                               'postclaainfo': postclaainfo,
                               'room': room, 'course': course, 'class_info': class_info})

                #     post.save()

        return render(request, 'teacheraddcourse_two.html',
                      {'teacher': teachers, 'timing': timing, 'semester': semester,
                       'postclaainfo': postclaainfo, 'room': room, 'course': course, 'class_info': class_info})


def student(request):
    if 'student_id' not in request.session:
        return redirect(index)
    else:
        student_id = request.session['student_id']
        students = Students.objects.get(id=student_id)

        context = {
            'student': students
        }

    return render(request, 'students.html', context)


def teacher(request):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)

        context = {
            'teacher': teachers
        }

    return render(request, 'teachers.html', context)


def errorImg(request):
    return render(request, 'error.html')


def create_dataset(request):
    # print request.POST
    userId = request.POST['userId']
    # print cv2.__version__
    # Detect face
    # Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')
    # camture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = userId
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while (True):
        # Capturing the image
        # cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        # the returned img is a colored image but for the classifier to work we need a greyscale image
        # to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # To store the faces
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        # In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for (x, y, w, h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum + 1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite(BASE_DIR + '/ml/dataset/user.' + str(id) + '.' + str(sampleNum) + '.jpg',
                        gray[y:y + h, x:x + w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.namedWindow('Face', cv2.WINDOW_NORMAL)
        cv2.imshow("Face", img)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        # To get out of the loop
        if (sampleNum > 35):
            break
    # releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

    return redirect(trainer)


def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    from PIL import Image

    # Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Path of the samples
    path = BASE_DIR + '/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # concatinate the path with the image name
        # print imagePaths

        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L')  # convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format
            ID = int(os.path.split(imagePath)[-1].split('.')[
                         1])  # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            # print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    # Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save(BASE_DIR + '/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect(student)


def detect(request, id):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)
        teacher_courses = Teacher_courses.objects.filter(teacher_id=teachers.id)
        face_records = []
        uniqueList = []
        faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')

        cam = cv2.VideoCapture(0)
        # creating recognizer
        rec = cv2.face.LBPHFaceRecognizer_create();
        # loading the training data
        rec.read(BASE_DIR + '/ml/recognizer/trainingData.yml')
        getId = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        userId = 0
        while (True):
            ret, img = cam.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                getId, conf = rec.predict(gray[y:y + h, x:x + w])  # This will predict the id of the face

                # print conf;
                if conf < 65:
                    userId = getId
                    cv2.putText(img, "Detected", (x, y + h), font, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Unknown", (x, y + h), font, 2, (0, 0, 255), 2)

                # Printing that number below the face
                # @Prams cam image, id, location,font style, color, stroke

            cv2.imshow("Face", img)
            if (userId != 0):
                cv2.waitKey(1000)

                face_records.append(userId)
                for face_id in face_records:
                    if face_id not in uniqueList:
                        uniqueList.append(face_id)

                print(uniqueList)
                # cam.release()
                # cv2.destroyAllWindows()
            if (cv2.waitKey(1) == ord('q')):
                # break
                print(uniqueList)

                student_course = Student_courses.objects.get(student_id=userId, teacher_course_id=id)
                # attandance = Attendance(student_course_id= Student_courses.objects.get(id=student_course.id),
                #                         created = datetime.today())
                # attandance.save()

                cam.release()
                cv2.destroyAllWindows()

                students = Students.objects.get(id=userId)

                student_courses = Student_courses.objects.filter(teacher_course_id=id)
                teacher_courses = Teacher_courses.objects.get(id=id)

                context = {"userId": userId,
                           "teacher": teachers,
                           # "teacher_courses":teacher_courses,
                           "student_courses": student_courses,
                           'teacher_course_id': id,
                           'teacher_courses': teacher_courses,
                           'uniqueList':uniqueList

                           }

                return render(request, 'coursestudentlist.html', context)

        cam.release()
        cv2.destroyAllWindows()
        return redirect('/coursestudentlist')


def submitattandance(request, id):
    if 'teacher_id' not in request.session:
        return redirect(index)
    else:
        teacher_id = request.session['teacher_id']
        teachers = Teachers.objects.get(id=teacher_id)

        student_courses = Student_courses.objects.filter(teacher_course_id=id)

        teacher_courses = Teacher_courses.objects.get(id=id)
        teacher_courses.class_count += 1
        teacher_courses.save()

        context = {

            'teacher': teachers,
            'student_courses': student_courses,
            'teacher_course_id': id,
            'teacher_courses': teacher_courses

        }

        return render(request, 'coursestudentlist.html', context)

