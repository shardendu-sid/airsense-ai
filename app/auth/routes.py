# import all necessary libraries

from app.Indoor_air_quality.routes import display_dashboard
import random
import string
from flask import render_template, redirect, flash, url_for

from app.auth.forms import RegistrationForm
from app.auth.forms import UserloginForm
from app.auth import authentication as at
from app.auth.models import User
from flask_login import login_user,logout_user, login_required, current_user
from flask import session,jsonify


# route signup form

@at.route('/signup', methods=['GET', 'POST'])
def signup():
    
    form = RegistrationForm()
    if current_user.is_authenticated:
        flash('You are already logged-in')
        return redirect(url_for('main.display_dashboard'))
    if form.validate_on_submit():
        User.create_user(
            user=form.name.data,
            email=form.email.data,
            password=form.password.data)
        flash('Signup Sucessfuly')
        return redirect(url_for('authentication.do_the_login'))

    return render_template('signup.html', form=form)


# route landing page 
# @at.route('/', methods=['GET', 'POST'])
# def do_the_login():
#     form = UserloginForm()
#     if current_user.is_authenticated:
#         flash('You are already logged-in')
#         return redirect(url_for('main.display_dashboard'))
    
#     if form.validate_on_submit():
#         user = User.query.filter_by(user_name=form.name.data).first()
#         if user is None:
#             flash('User not found')
#             return redirect(url_for('authentication.do_the_login'))

#         if not user.check_password(form.password.data):
#             flash('Invalid Credentials, Please try again')
#             return redirect(url_for('authentication.do_the_login'))

#         login_user(user, form.remember_me.data)
#         session.permanent = True  # Activate session timeout management
#         return redirect(url_for('main.display_dashboard'))
    
#     flash('Form not submitted correctly')
#     return render_template('login.html', form=form)

def rand_pass(length=6):
    generate_pass=''.join([random.choice(string.ascii_uppercase+string.ascii_lowercase+string.digits)for n in range(length)])    
    return generate_pass 


@at.route('/', methods=['GET', 'POST'])
def do_the_login():
    form = UserloginForm()
    if current_user.is_authenticated:
        flash('You are already logged-in')
        return redirect(url_for('main.display_dashboard'))
    
    # Generate an OTP and store it in session for verification
    if 'rand_pass' not in session:
        session['rand_pass'] = rand_pass()

    if form.validate_on_submit():
        user = User.query.filter_by(user_name=form.name.data).first()

        if user and form.password.data == session.get('rand_pass'):
            login_user(user)  # Log in the user
            session.pop('rand_pass', None)
            # flash('Invalid Credentials, Please try again')
            return redirect(url_for('main.display_dashboard'))
        else:
            flash('Invalid username or OTP')
            return redirect(url_for('authentication.do_the_login'))



        # login_user(user, form.remember_me.data)
        # session.permanent = True  
        # return redirect(url_for('main.display_dashboard'))
    return render_template('login.html', form=form,rand_pass=session['rand_pass'])

# route lagout page

@at.route('/logout')
@login_required
def log_out_user():
    logout_user()
    flash('you are loged-out')
    return redirect(url_for('authentication.do_the_login'))


# continue session
@at.route('/keep-alive', methods=['GET'])
@login_required
def keep_alive():
    session.modified = True  # This refreshes the session timeout
    return jsonify(success=True)