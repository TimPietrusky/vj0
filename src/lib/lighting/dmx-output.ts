/**
 * DmxOutput - WebUSB DMX512 controller wrapper
 *
 * Manages connection to a WebUSB DMX512 controller (Arduino Leonardo-based)
 * and sends DMX universe data to the device.
 *
 * Based on webusb-dmx512-controller by NERDDISCO.
 */

import type { DmxUniverse } from "./types";

// USB device filters for supported Arduino boards
const USB_FILTERS = [
  // Arduino Leonardo
  { vendorId: 0x2341, productId: 0x8036 },
  { vendorId: 0x2341, productId: 0x0036 },
  { vendorId: 0x2a03, productId: 0x8036 },
  { vendorId: 0x2a03, productId: 0x0036 },
  // Arduino Leonardo ETH
  { vendorId: 0x2a03, productId: 0x0040 },
  { vendorId: 0x2a03, productId: 0x8040 },
  // Seeeduino Lite
  { vendorId: 0x2886, productId: 0x8002 },
];

export class DmxOutput {
  private device: USBDevice | null = null;
  private connected = false;

  /**
   * Check if WebUSB is supported in this browser
   */
  static isSupported(): boolean {
    return typeof navigator !== "undefined" && "usb" in navigator;
  }

  /**
   * Find a previously paired USB device that matches one of our supported
   * Arduino vendor/product combos. Filtering matters: this origin can have
   * other unrelated paired devices, and `getDevices()[0]` would pick one of
   * those at random — claimInterface(2) would then fail silently and the
   * user sees "auto-connect doesn't work" with no obvious cause.
   */
  private async getPairedDevice(): Promise<USBDevice | undefined> {
    if (!DmxOutput.isSupported()) return undefined;
    const devices = await navigator.usb.getDevices();
    return devices.find((d) =>
      USB_FILTERS.some(
        (f) => f.vendorId === d.vendorId && f.productId === d.productId
      )
    );
  }

  /**
   * Open connection to a USB device
   */
  private async openDevice(device: USBDevice): Promise<void> {
    // Open connection
    await device.open();

    // Select configuration if not auto-selected
    if (device.configuration === null) {
      await device.selectConfiguration(1);
    }

    // Claim interface #2
    await device.claimInterface(2);

    // Signal that we're ready to send data (SET_CONTROL_LINE_STATE)
    await device.controlTransferOut({
      requestType: "class",
      recipient: "interface",
      request: 0x22,
      value: 0x01,
      index: 0x02,
    });

    this.device = device;
    this.connected = true;
  }

  /**
   * Automatically connect to a previously paired USB device
   * @returns true if connected, false if no paired device found
   */
  async autoConnect(): Promise<boolean> {
    if (!DmxOutput.isSupported()) return false;

    try {
      const device = await this.getPairedDevice();
      if (!device) return false;

      await this.openDevice(device);
      return true;
    } catch (err) {
      // Surface the failure — silent catches make "auto-connect doesn't work"
      // impossible to diagnose. Common causes: another tab holds the USB
      // claim, or the device disappeared mid-open.
      console.warn("[DMX] autoConnect failed:", err);
      this.device = null;
      this.connected = false;
      return false;
    }
  }

  /**
   * Open the WebUSB device picker and connect to the selected device
   */
  async connect(): Promise<void> {
    if (!DmxOutput.isSupported()) {
      throw new Error("WebUSB is not supported in this browser");
    }

    try {
      // Request device from user
      const device = await navigator.usb.requestDevice({ filters: USB_FILTERS });
      await this.openDevice(device);
    } catch (err) {
      this.device = null;
      this.connected = false;
      throw err;
    }
  }

  /**
   * Disconnect from the USB device
   */
  async disconnect(): Promise<void> {
    if (!this.device) return;

    try {
      // Send all zeros to turn off fixtures
      const blackout = new Uint8Array(512);
      await this.sendUniverse(blackout);

      // Signal that we're done
      await this.device.controlTransferOut({
        requestType: "class",
        recipient: "interface",
        request: 0x22,
        value: 0x00,
        index: 0x02,
      });

      // Close connection
      await this.device.close();
    } catch {
      // Ignore errors during disconnect
    } finally {
      this.device = null;
      this.connected = false;
    }
  }

  /**
   * Check if currently connected to a DMX device
   */
  isConnected(): boolean {
    return this.connected && this.device !== null;
  }

  /**
   * Get device info if connected
   */
  getDeviceInfo(): { productName: string; manufacturerName: string } | null {
    if (!this.device) return null;
    return {
      productName: this.device.productName || "Unknown",
      manufacturerName: this.device.manufacturerName || "Unknown",
    };
  }

  /**
   * Send the entire DMX universe to the device
   */
  async sendUniverse(universe: DmxUniverse): Promise<void> {
    if (!this.device || !this.connected) return;

    try {
      // Send data on Endpoint #4
      await this.device.transferOut(4, universe);
    } catch {
      // Connection may have been lost
      this.connected = false;
    }
  }
}

