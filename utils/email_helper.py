import smtplib
from email.message import EmailMessage

def send_email(to_email, subject, attachment_path):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "your_email@gmail.com"
    msg['To'] = to_email
    msg.set_content("Attached is your NASA Climate NLP report.")

    with open(attachment_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='application', subtype='csv', filename='nasa_topic_output.csv')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login("your_email@gmail.com", "your_app_password")
        smtp.send_message(msg)
