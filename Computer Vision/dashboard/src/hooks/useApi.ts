import { useState, useEffect, useCallback, useRef } from 'react';
import { checkBackendAvailable, getSystemInfo } from '../services/api';
import type { SystemInfo } from '../types';

export function useBackendStatus() {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(true);

  const check = useCallback(async () => {
    setChecking(true);
    const ok = await checkBackendAvailable();
    setAvailable(ok);
    setChecking(false);
  }, []);

  useEffect(() => { check(); }, [check]);

  return { available, checking, recheck: check };
}

export function useSystemInfo() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSystemInfo()
      .then(setInfo)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { info, loading, error };
}

export function useApiCall<T>() {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef(false);

  const execute = useCallback(async (fn: () => Promise<T>): Promise<{ data: T | null; error: string | null }> => {
    cancelRef.current = false;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await fn();
      if (!cancelRef.current) {
        setData(result);
      }
      return { data: result, error: null };
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!cancelRef.current) {
        setError(msg);
      }
      return { data: null, error: msg };
    } finally {
      if (!cancelRef.current) {
        setLoading(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    cancelRef.current = true;
    setData(null);
    setLoading(false);
    setError(null);
  }, []);

  return { data, loading, error, execute, reset };
}
