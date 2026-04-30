/**
 * Minimal WebUSB type augmentation. Lets `navigator.usb` typecheck without
 * pulling in @types/w3c-web-usb (which the project doesn't depend on). The
 * runtime `dmx-output.ts` driver uses the full USBDevice surface — this
 * declaration covers the additional `navigator.usb` access points that
 * VJApp uses for hotplug listening.
 *
 * If we ever need stricter typing we should switch to the upstream package.
 * For now we lean on `unknown` everywhere so callers must narrow.
 */

interface USB extends EventTarget {
  getDevices(): Promise<USBDevice[]>;
  requestDevice(options: { filters: { vendorId?: number; productId?: number }[] }): Promise<USBDevice>;
  addEventListener(type: "connect" | "disconnect", listener: (event: Event) => void): void;
  removeEventListener(type: "connect" | "disconnect", listener: (event: Event) => void): void;
}

interface USBDevice {
  readonly vendorId: number;
  readonly productId: number;
  open(): Promise<void>;
  close(): Promise<void>;
  selectConfiguration(configurationValue: number): Promise<void>;
  claimInterface(interfaceNumber: number): Promise<void>;
  releaseInterface(interfaceNumber: number): Promise<void>;
  controlTransferOut(setup: unknown, data?: BufferSource): Promise<unknown>;
  transferOut(endpointNumber: number, data: BufferSource): Promise<unknown>;
}

interface Navigator {
  readonly usb: USB;
}
