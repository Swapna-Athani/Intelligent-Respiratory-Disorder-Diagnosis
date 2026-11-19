"""Firebase client REST only. Every database request uses the caller's ID token."""
import re
import requests
from config import API_KEY,DATABASE_URL,ROOT
class FirebaseError(Exception): pass

def identity(token):
    try:
        response=requests.post('https://identitytoolkit.googleapis.com/v1/accounts:lookup',params={'key':API_KEY},json={'idToken':token},timeout=15)
        if response.status_code!=200: raise FirebaseError('Sign in again; the Firebase session is invalid or expired.')
        uid=response.json()['users'][0]['localId']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',uid): raise FirebaseError('Unsupported user identifier.')
        return uid
    except (requests.RequestException,KeyError,IndexError,ValueError):
        raise FirebaseError('Unable to verify the Firebase session.') from None

def database(method,uid,token,path='',payload=None):
    if not re.fullmatch(r'[A-Za-z0-9_/-]*',path): raise FirebaseError('Invalid record path.')
    try:
        response=requests.request(method,f'{DATABASE_URL}/{ROOT}/users/{uid}/{path}.json',params={'auth':token},json=payload,timeout=20)
        if response.status_code>=400: raise FirebaseError('Firebase access denied or unavailable. Check your session and project database rules.')
        return response.json()
    except (requests.RequestException,ValueError): raise FirebaseError('Firebase request failed; no result was confirmed saved.') from None
