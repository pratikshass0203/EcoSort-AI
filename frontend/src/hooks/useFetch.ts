// src/hooks/useFetch.ts
import { useState, useEffect } from 'react';

export const useFetch = <T>(url: string) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    const fetchData = async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Error ${res.status}`);
        const json = (await res.json()) as T;
        if (!ignore) setData(json);
      } catch (e:any) {
        if (!ignore) setError(e.message);
      } finally {
        if (!ignore) setLoading(false);
      }
    };
    fetchData();
    return () => { ignore = true; };
  }, [url]);

  return { data, loading, error };
};

