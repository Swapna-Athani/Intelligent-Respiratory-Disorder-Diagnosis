import React, {useState} from 'react';
import {browserSessionPersistence, createUserWithEmailAndPassword, setPersistence, signInWithEmailAndPassword} from 'firebase/auth';
import {auth} from './firebase';

function authMessage(error) {
  switch (error.code) {
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found': return 'The email or password is incorrect. Please try again.';
    case 'auth/email-already-in-use': return 'An account already uses this email. Switch to Log in to continue.';
    case 'auth/invalid-email': return 'Enter a valid email address.';
    case 'auth/weak-password': return 'Choose a stronger password with at least 8 characters.';
    case 'auth/password-does-not-meet-requirements': return 'This password does not meet the account requirements. Choose a stronger password.';
    case 'auth/too-many-requests': return 'Too many attempts. Please wait a little before trying again.';
    case 'auth/network-request-failed': return 'Could not connect. Check your internet connection and try again.';
    case 'auth/operation-not-allowed': return 'Email/password accounts are not enabled in this Firebase project. Enable Email/Password in Firebase Authentication.';
    case 'auth/unauthorized-domain': return 'This website address must be added to the Firebase authorized domains.';
    default: return 'Unable to complete authentication. Please try again.';
  }
}

export default function AuthScreen() {
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [visible, setVisible] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const registering = mode === 'register';

  function switchMode(next) {
    setMode(next);
    setPassword('');
    setConfirmation('');
    setVisible(false);
    setError('');
  }

  async function submit(event) {
    event.preventDefault();
    if (pending) return;
    setError('');
    if (registering && password.length < 8) {
      setError('Choose a password with at least 8 characters.');
      return;
    }
    if (registering && password !== confirmation) {
      setError('Passwords do not match. Re-enter your confirmation password.');
      return;
    }
    setPending(true);
    try {
      await setPersistence(auth, browserSessionPersistence);
      if (registering) {
        await createUserWithEmailAndPassword(auth, email.trim(), password);
      } else {
        await signInWithEmailAndPassword(auth, email.trim(), password);
      }
      setPassword('');
      setConfirmation('');
      // App opens the dashboard only after Firebase confirms the authenticated user.
    } catch (err) {
      setError(authMessage(err));
    } finally {
      setPending(false);
    }
  }

  return <div className="auth-screen">
    <section className="auth-intro" aria-labelledby="auth-brand">
      <div className="auth-brand">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M10 3v9c-2-1-2-7-4-6S1 14 2 19s8 1 8-2M14 3v9c2-1 2-7 4-6s5 8 4 13-8 1-8-2M12 1v8m0 0-4 4m4-4 4 4"/></svg>
        <span id="auth-brand">Respira Monitor</span>
      </div>
      <div className="auth-intro-copy">
        <span className="auth-eyebrow">RESPIRATORY SCREENING</span>
        <h1>A clearer view of<br/>respiratory health.</h1>
        <p>Bring lung sound and chest X-ray analysis together in one workspace.</p>
        <div className="auth-features">
          <div><span aria-hidden="true">01</span><div><strong>Lung sound analysis</strong><p>Review recordings, waveforms and spectrograms.</p></div></div>
          <div><span aria-hidden="true">02</span><div><strong>Chest X-ray screening</strong><p>Upload images and review model results.</p></div></div>
          <div><span aria-hidden="true">03</span><div><strong>Your test workspace</strong><p>Access saved tests and generate screening reports.</p></div></div>
        </div>
      </div>
      <p className="auth-intro-note">A preliminary screening prototype. Professional medical interpretation is required.</p>
    </section>
    <section className="auth-form-side" aria-labelledby="auth-title">
      <div className="auth-card">
        <div className="auth-mode" aria-label="Account access">
          <button type="button" aria-pressed={!registering} className={!registering ? 'active' : ''} disabled={pending} onClick={() => switchMode('login')}>Log in</button>
          <button type="button" aria-pressed={registering} className={registering ? 'active' : ''} disabled={pending} onClick={() => switchMode('register')}>Register</button>
        </div>
        <span className="auth-eyebrow">{registering ? 'GET STARTED' : 'YOUR WORKSPACE AWAITS'}</span>
        <h2 id="auth-title">{registering ? 'Create your account' : 'Welcome back'}</h2>
        <p className="auth-subtitle">{registering ? 'Register to start reviewing respiratory samples and managing your tests.' : 'Log in to access your respiratory screening dashboard.'}</p>
        <form onSubmit={submit} aria-busy={pending}>
          <label htmlFor="auth-email">Email address</label>
          <input id="auth-email" name="email" type="email" autoComplete="username" placeholder="you@example.com" required maxLength={254} disabled={pending} value={email} onChange={e => setEmail(e.target.value)}/>
          <div className="auth-password-heading"><label htmlFor="auth-password">Password</label><button type="button" aria-controls="auth-password" aria-pressed={visible} disabled={pending} onClick={() => setVisible(!visible)}>{visible ? 'Hide passwords' : 'Show passwords'}</button></div>
          <input id="auth-password" name="password" type={visible ? 'text' : 'password'} autoComplete={registering ? 'new-password' : 'current-password'} required minLength={registering ? 8 : undefined} disabled={pending} aria-describedby={registering ? 'auth-password-hint' : undefined} value={password} onChange={e => setPassword(e.target.value)}/>
          {registering && <><small id="auth-password-hint" className="auth-field-hint">Use at least 8 characters.</small><label htmlFor="auth-confirm">Confirm password</label><input id="auth-confirm" name="confirmPassword" type={visible ? 'text' : 'password'} autoComplete="new-password" required minLength={8} disabled={pending} value={confirmation} onChange={e => setConfirmation(e.target.value)}/></>}
          {error && <p role="alert" className="error auth-error">{error}</p>}
          <button type="submit" className="auth-submit" disabled={pending}>{pending ? (registering ? 'Creating account...' : 'Logging in...') : (registering ? 'Create account' : 'Log in')}<span aria-hidden="true">&rarr;</span></button>
          {pending && <p className="auth-progress" role="status">{registering ? 'Creating your account. Please wait.' : 'Signing you in. Please wait.'}</p>}
        </form>
        <p className="auth-switch">{registering ? 'Already have an account?' : 'New to Respira Monitor?'} <button type="button" disabled={pending} onClick={() => switchMode(registering ? 'login' : 'register')}>{registering ? 'Log in' : 'Create an account'}</button></p>
        <p className="auth-session-note">Your sign-in is kept for this browser session.</p>
      </div>
    </section>
  </div>;
}
