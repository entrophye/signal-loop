 import json, os, sys
 from google_auth_oauthlib.flow import InstalledAppFlow
 from google.auth.transport.requests import Request
 from google.oauth2.credentials import Credentials
 SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
 # Provide client_id/client_secret via input or a local JSON
 CLIENT_ID = os.getenv('YOUTUBE_CLIENT_ID') or input('Client ID: ').strip()
 CLIENT_SECRET = os.getenv('YOUTUBE_CLIENT_SECRET') or input('Client Secret: 
').strip()
 client_config = {
 "installed": {
 "client_id": CLIENT_ID,
 "client_secret": CLIENT_SECRET,
 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
 "token_uri": "https://oauth2.googleapis.com/token",
 "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob","http://localhost"]
 }
 }
 flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
 creds = flow.run_console()
 print('\nREFRESH TOKEN:')
 print(creds.refresh_token)
