import React, {useState} from 'react';
import {updateProfile} from 'firebase/auth';
import {auth} from './firebase';

function dateLabel(value) {
  if (!value) return 'Not available';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Not available' : date.toLocaleString();
}

export default function Profile({user, name, onSaved}) {
  const [displayName, setDisplayName] = useState(name);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  async function save(event) {
    event.preventDefault();
    if (saving) return;
    setError(''); setMessage('');
    const nextName = displayName.trim();
    if (!nextName) {setError('Enter your full name.'); return;}
    if (!auth.currentUser || auth.currentUser.uid !== user.uid) {
      setError('Your session has changed. Please log in again.'); return;
    }
    setSaving(true);
    try {
      await updateProfile(auth.currentUser, {displayName: nextName});
      if (auth.currentUser?.uid === user.uid) {
        onSaved(nextName);
        setDisplayName(nextName);
        setMessage('Your profile has been saved.');
      }
    } catch (err) {
      setError(err.code === 'auth/network-request-failed' ? 'Could not connect. Check your internet connection and try again.' : 'Could not save your profile. Please try again.');
    } finally {setSaving(false);}
  }
  return <section className="card profile-section" aria-labelledby="profile-title">
    <div className="profile-heading"><span className="profile-avatar" aria-hidden="true">{(name || user.email || 'U').slice(0, 1).toUpperCase()}</span><div><h2 id="profile-title">Your profile</h2><p>Manage the name shown in your workspace.</p></div></div>
    <form onSubmit={save} aria-busy={saving} className="profile-form">
      <label htmlFor="profile-name">Full name</label>
      <input id="profile-name" name="name" autoComplete="name" value={displayName} maxLength={100} required disabled={saving} onChange={event => setDisplayName(event.target.value)}/>
      <label htmlFor="profile-email">Email address</label>
      <input id="profile-email" type="email" value={user.email || ''} readOnly/>
      <small>Your email is the address used to log in.</small>
      {error && <p role="alert" className="error">{error}</p>}
      {message && <p role="status" className="message">{message}</p>}
      <button disabled={saving} type="submit">{saving ? 'Saving profile...' : 'Save profile'}</button>
    </form>
    <dl className="profile-details"><div><dt>Account created</dt><dd>{dateLabel(user.metadata?.creationTime)}</dd></div><div><dt>Last login</dt><dd>{dateLabel(user.metadata?.lastSignInTime)}</dd></div><div><dt>Email status</dt><dd>{user.emailVerified ? 'Verified' : 'Not verified'}</dd></div></dl>
  </section>;
}
