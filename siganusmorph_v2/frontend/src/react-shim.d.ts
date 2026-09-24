declare module "react" {
  export type SetStateAction<S> = S | ((prevState: S) => S);
  export type Dispatch<A> = (value: A) => void;
  export type RefObject<T> = { current: T };

  export type PointerEvent<T = Element> = {
    currentTarget: T;
    clientX: number;
    clientY: number;
    pointerId: number;
  };

  export type KeyboardEvent<T = Element> = {
    currentTarget: T;
    key: string;
    shiftKey: boolean;
    preventDefault: () => void;
  };

  export type ChangeEvent<T = Element> = {
    target: T;
  };

  export const StrictMode: (props: { children?: unknown }) => JSX.Element;
  export function useState<S>(initialState: S | (() => S)): [S, Dispatch<SetStateAction<S>>];
  export function useEffect(effect: () => void | (() => void), deps?: readonly unknown[]): void;
  export function useMemo<T>(factory: () => T, deps: readonly unknown[]): T;
  export function useRef<T>(initialValue: T): RefObject<T>;
}

declare module "react/jsx-runtime" {
  export const Fragment: unknown;
  export function jsx(type: unknown, props: unknown, key?: unknown): unknown;
  export function jsxs(type: unknown, props: unknown, key?: unknown): unknown;
}

declare module "react-dom/client" {
  export function createRoot(container: Element | DocumentFragment): {
    render: (node: unknown) => void;
  };
}

declare namespace JSX {
  type Element = any;
  interface IntrinsicElements {
    [elementName: string]: any;
  }
}
