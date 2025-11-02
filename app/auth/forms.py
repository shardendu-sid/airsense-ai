from app.auth.models import User
from flask_wtf import FlaskForm
# from flask_wtf.recaptcha import validators
from wtforms import StringField, SubmitField, PasswordField, BooleanField
from wtforms.fields.simple import PasswordField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


def email_exists(form,field):
    email = User.query.filter_by(user_email=field.data).first()
    if email:
        raise ValidationError('Name Already Exists')


# create user registration form

class RegistrationForm(FlaskForm):
    name = StringField('Username', validators=[DataRequired(), Length(3,15, message='between 3 to 15 characters')])
    email = StringField('Email', validators=[DataRequired(), Email(), email_exists])
    password = PasswordField('Password', validators=[DataRequired()])


# create user login form


class UserloginForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(3,15, message='between 3 to 15 characters')])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember me')

