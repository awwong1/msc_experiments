//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:37 PM EDT 
//


package org.sierraecg.schema.jaxb._1_03;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element ref="{http://www3.medical.philips.com}parsedwaveforms" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}unparsedwaveforms" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}leadwaveforms" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}vcgs" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}repbeats" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "parsedwaveforms",
    "unparsedwaveforms",
    "leadwaveforms",
    "vcgs",
    "repbeats"
})
@XmlRootElement(name = "waveforms")
public class Waveforms {

    protected Parsedwaveforms parsedwaveforms;
    protected Unparsedwaveforms unparsedwaveforms;
    protected Leadwaveforms leadwaveforms;
    protected Vcgs vcgs;
    protected Repbeats repbeats;

    /**
     * Gets the value of the parsedwaveforms property.
     * 
     * @return
     *     possible object is
     *     {@link Parsedwaveforms }
     *     
     */
    public Parsedwaveforms getParsedwaveforms() {
        return parsedwaveforms;
    }

    /**
     * Sets the value of the parsedwaveforms property.
     * 
     * @param value
     *     allowed object is
     *     {@link Parsedwaveforms }
     *     
     */
    public void setParsedwaveforms(Parsedwaveforms value) {
        this.parsedwaveforms = value;
    }

    /**
     * Gets the value of the unparsedwaveforms property.
     * 
     * @return
     *     possible object is
     *     {@link Unparsedwaveforms }
     *     
     */
    public Unparsedwaveforms getUnparsedwaveforms() {
        return unparsedwaveforms;
    }

    /**
     * Sets the value of the unparsedwaveforms property.
     * 
     * @param value
     *     allowed object is
     *     {@link Unparsedwaveforms }
     *     
     */
    public void setUnparsedwaveforms(Unparsedwaveforms value) {
        this.unparsedwaveforms = value;
    }

    /**
     * Gets the value of the leadwaveforms property.
     * 
     * @return
     *     possible object is
     *     {@link Leadwaveforms }
     *     
     */
    public Leadwaveforms getLeadwaveforms() {
        return leadwaveforms;
    }

    /**
     * Sets the value of the leadwaveforms property.
     * 
     * @param value
     *     allowed object is
     *     {@link Leadwaveforms }
     *     
     */
    public void setLeadwaveforms(Leadwaveforms value) {
        this.leadwaveforms = value;
    }

    /**
     * Gets the value of the vcgs property.
     * 
     * @return
     *     possible object is
     *     {@link Vcgs }
     *     
     */
    public Vcgs getVcgs() {
        return vcgs;
    }

    /**
     * Sets the value of the vcgs property.
     * 
     * @param value
     *     allowed object is
     *     {@link Vcgs }
     *     
     */
    public void setVcgs(Vcgs value) {
        this.vcgs = value;
    }

    /**
     * Gets the value of the repbeats property.
     * 
     * @return
     *     possible object is
     *     {@link Repbeats }
     *     
     */
    public Repbeats getRepbeats() {
        return repbeats;
    }

    /**
     * Sets the value of the repbeats property.
     * 
     * @param value
     *     allowed object is
     *     {@link Repbeats }
     *     
     */
    public void setRepbeats(Repbeats value) {
        this.repbeats = value;
    }

}
