import os
from socket import gaierror
from smtplib import SMTP_SSL
from ssl import create_default_context


def email_alert(message: str = 'Subject: Python alert\n\nThis is an email alert sent from a Python script!',
                to: str = 'simpyalerts@gmail.com'):
    """Send an email with a given message to a given valid address."""
    try:
        # Create a secure SSL context and use it to send an email from a "throwaway" account to inform that the
        # script has finished.
        with SMTP_SSL(host="smtp.gmail.com", port=465, context=create_default_context()) as server:
            server.login(user='simpyalerts@gmail.com', password='$impyAlerts12345')
            server.sendmail('simpyalerts@gmail.com', to, message)
    except gaierror:
        print('\nEmail alert could not be sent. No internet connection available!\n')


def sound_alert(frequency: float = 440, duration: float = 2):
    """Reproduce a sound with a given frequency (in Hz) and for a given duration (in seconds)."""
    command = 'play -nq -t alsa synth {} sine {}'.format(duration, frequency)
    result = os.system(command)
    if result != 0:
        print('\nSound alert could not be reproduced. A wrong command was passed or sox package is not available!\n')


def speech_alert(message: str = 'Your program has finished'):
    """Reproduce a speech with a given message."""
    command = 'spd-say "{}"'.format(message)
    result = os.system(command)
    if result != 0:
        print('\nSpeech alert could not be reproduced. A wrong command was passed or speech-dispatcher package is '
              'not available!\n')
