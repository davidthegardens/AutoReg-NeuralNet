event = {
  'summary': 'Content + Meme Session (please attent)',
  'location': 'Rex Hygate',
  'description': 'Meme and content for de fi safty',
  'start': {
    'dateTime': '2015-05-28T09:00:00-07:00',
    'timeZone': 'America/Los_Angeles',
  },
  'end': {
    'dateTime': '2015-05-28T17:00:00-07:00',
    'timeZone': 'America/Los_Angeles',
  },
  'recurrence': [
    'RRULE:FREQ=WEEKLY'
  ],
  'attendees': [
    {'email': 'ryomamartin@hotmail.com'},
    {'email': 'ryoma@defisafety.com'},
  ],
  'reminders': {
    'useDefault': False,
    'overrides': [
      {'method': 'email', 'minutes': 24 * 60},
      {'method': 'popup', 'minutes': 10},
    ],
  },
}

event = service.events().insert(calendarId='primary', body=event).execute()
