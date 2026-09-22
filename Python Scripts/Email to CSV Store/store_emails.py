import argparse
import csv
import email
from email import policy
import imaplib
import logging
from pathlib import Path
import ssl

from bs4 import BeautifulSoup


logger = logging.getLogger('imap_poller')

HOST = 'imap.gmail.com'
port = 993
ssl_context = ssl.create_default_context()


def connect_to_mailbox(credential_path: Path):
    mail = imaplib.IMAP4_SSL(HOST, port, ssl_context=ssl_context)

    with credential_path.open('rt', encoding='utf-8') as fr:
        user = fr.readline().strip()
        pw = fr.readline().strip()

    if not user or not pw:
        raise ValueError('The credentials file must contain an email address and app password.')

    mail.login(user, pw)

    status, messages = mail.select('INBOX')
    if status != 'OK':
        raise imaplib.IMAP4.error('Unable to select INBOX.')

    return mail, messages


# get plain text out of html mails
def get_text(email_body: str) -> str:
    soup = BeautifulSoup(email_body, 'lxml')
    return soup.get_text(separator='\n', strip=True)


def write_to_csv(mail, writer, count: int, total_no_of_mails: int) -> None:

    for i in range(total_no_of_mails, max(total_no_of_mails - count, 0), -1):
        res, data = mail.fetch(str(i), '(RFC822)')

        response = data[0]
        if isinstance(response, tuple):
            msg = email.message_from_bytes(response[1], policy=policy.default)

            # get header data
            email_subject = msg["subject"]
            email_from = msg["from"]
            email_date = msg["date"]
            email_text = ""

            # if the email message is multipart
            if msg.is_multipart():
                # iterate over email parts
                for part in msg.walk():
                    # extract content type of email
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    try:
                        # get the email email_body
                        email_body = part.get_payload(decode=True)
                        if email_body:
                            email_text = get_text(email_body.decode('utf-8'))
                    except Exception as exc:
                        logger.warning('Caught exception: %r', exc)
                    if (
                        content_type == "text/plain"
                        and "attachment" not in content_disposition
                    ):
                        # print text/plain emails and skip attachments
                        # print(email_text)
                        pass
                    elif "attachment" in content_disposition:
                        pass

            else:
                # extract content type of email
                content_type = msg.get_content_type()
                # get the email email_body
                email_body = msg.get_payload(decode=True)
                if email_body:
                    email_text = get_text(email_body.decode('utf-8'))

            if email_text is not None:
                # Write data in the csv file
                row = [email_date, email_from, email_subject, email_text]
                writer.writerow(row)
            else:
                logger.warning('%s:%i: No message extracted', "INBOX", i)

def main() -> None:
    parser = argparse.ArgumentParser(description='Export recent Gmail inbox messages to CSV.')
    parser.add_argument('--count', type=int, default=2, help='Number of recent messages to export')
    parser.add_argument(
        '--credentials',
        type=Path,
        default=Path(__file__).with_name('credentials.txt'),
        help='File with the email address on line 1 and app password on line 2',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).with_name('mails.csv'),
        help='CSV file to create',
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be at least 1')

    logging.basicConfig(level=logging.WARNING)

    mail, messages = connect_to_mailbox(args.credentials)
    try:
        total_no_of_mails = int(messages[0])
        with args.output.open('w', encoding='utf-8', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(['Date', 'From', 'Subject', 'Text mail'])
            write_to_csv(mail, writer, args.count, total_no_of_mails)
    finally:
        mail.logout()


if __name__ == "__main__":
    main()
