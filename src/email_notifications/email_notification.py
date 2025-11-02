import os
import smtplib # SMTP client session
from email.message import EmailMessage # for email notification
import ssl # for security
import bcrypt

email_notification = os.environ.get('email')
email_notification_password = os.environ.get('email_password')

def send_email_notification_turn_on():

            email_sender = email_notification
            email_password = email_notification_password

            email_receiver = email_notification

            subject = 'Awair sensor notification'
            body = """
                    Subject: Awair sensor temperature notification!

                    Your room temperature has exceeded the threshold, and the air cooler has turned on successfully.

                    Regards,
                    Test Company,
                    Vantaa, Finland
            """

            em = EmailMessage()
            em['From'] = email_sender 
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)

            context = ssl.create_default_context()

            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(email_sender,email_password)
                smtp.sendmail(email_sender,email_receiver,em.as_string())  # Send email


def send_email_notification_turn_off():

    email_sender = email_notification
    email_password = email_notification_password 
    email_receiver = email_notification

    subject = 'Awair sensor notification'
    body = """
            Subject: Awair sensor temperature notification!

            Your room temperature has exceeded the threshold, and the air cooler has turned off successfully.

            Regards,
            Test Company,
            Vantaa, Finland
    """

    em = EmailMessage()
    em['From'] = email_sender 
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender,email_password)
        smtp.sendmail(email_sender,email_receiver,em.as_string())  # Send email

